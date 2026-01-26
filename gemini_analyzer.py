# gemini_analyzer.py
"""
Gemini-based analyzer for nightly processing of chat messages.
Analyzes daily messages and builds/updates user knowledge graphs.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import google.generativeai as genai

from models import BotConfig, ChatMessage
from knowledge_graph import UserKnowledgeGraph, KnowledgeGraphManager, InterestNode

logger = logging.getLogger(__name__)


# Gemini prompt for analyzing messages and building knowledge graph
GEMINI_ANALYSIS_PROMPT = """Ты анализатор сообщений чата. Твоя задача - извлечь информацию о пользователях из их сообщений.

Проанализируй следующие сообщения пользователя {username} за день и создай/обнови его профиль.

СООБЩЕНИЯ:
{messages}

Верни JSON в ТОЧНО таком формате (без markdown, только чистый JSON):
{{
    "quick_facts": ["короткий факт 1", "короткий факт 2"],
    "interests": {{
        "gaming": {{
            "название_игры": {{"role": "роль", "frequency": "частота"}}
        }},
        "food": {{
            "предпочтение": {{"type": "тип", "frequency": "частота"}}
        }},
        "education": {{
            "предмет": {{"level": "уровень"}}
        }},
        "work": {{
            "сфера": {{"position": "должность"}}
        }},
        "entertainment": {{
            "тип": {{"preference": "предпочтение"}}
        }},
        "tech": {{
            "технология": {{"level": "уровень"}}
        }},
        "sports": {{
            "вид_спорта": {{"frequency": "частота"}}
        }},
        "music": {{
            "жанр_или_артист": {{"preference": "предпочтение"}}
        }}
    }},
    "personal": {{
        "mood": "общее настроение",
        "communication_style": "стиль общения"
    }},
    "social": {{
        "friends_mentioned": ["имена друзей если упоминались"],
        "group_role": "роль в группе"
    }},
    "patterns": {{
        "typical_topics": ["тема1", "тема2"],
        "active_hours": [час1, час2]
    }}
}}

ПРАВИЛА:
1. Заполняй ТОЛЬКО те поля, для которых есть информация в сообщениях
2. Пустые категории оставляй как пустые объекты {{}}
3. quick_facts - максимум 5 коротких фактов
4. Пиши на русском языке
5. Не выдумывай информацию - только то, что явно следует из сообщений
6. Верни ТОЛЬКО JSON, без пояснений"""


class GeminiAnalyzer:
    """
    Analyzes chat messages using Gemini API to build knowledge graphs.
    Designed to run nightly at 3:00 AM.
    """
    
    def __init__(self, api_key: str, knowledge_manager: KnowledgeGraphManager):
        """
        Initialize Gemini analyzer.
        
        Args:
            api_key: Google Gemini API key
            knowledge_manager: KnowledgeGraphManager instance
        """
        self._knowledge_manager = knowledge_manager
        
        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel('gemini-pro')
            logger.info("GeminiAnalyzer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    async def analyze_user_messages(
        self, 
        user_id: int, 
        username: str,
        messages: List[ChatMessage]
    ) -> Optional[UserKnowledgeGraph]:
        """
        Analyze messages for a single user and update their knowledge graph.
        
        Args:
            user_id: Telegram user ID
            username: Username
            messages: List of user's messages from the day
            
        Returns:
            Updated UserKnowledgeGraph or None on error
        """
        if not messages:
            logger.debug(f"No messages to analyze for user {user_id}")
            return None
        
        # Format messages for Gemini
        messages_text = "\n".join([
            f"[{msg.timestamp.strftime('%H:%M')}] {msg.text}"
            for msg in messages
        ])
        
        prompt = GEMINI_ANALYSIS_PROMPT.format(
            username=username,
            messages=messages_text
        )
        
        try:
            # Call Gemini API
            response = self._model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Clean up response (remove markdown if present)
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            # Parse JSON response
            analysis = json.loads(response_text)
            
            # Get or create user's knowledge graph
            graph = self._knowledge_manager.get_user_graph(user_id, username)
            
            # Update graph with analysis results
            self._update_graph_from_analysis(graph, analysis)
            
            # Save updated graph
            self._knowledge_manager.save_user_graph(graph)
            
            logger.info(f"Successfully analyzed {len(messages)} messages for user {username}")
            return graph
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.debug(f"Response was: {response_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Error analyzing messages for user {user_id}: {e}")
            return None
    
    def _update_graph_from_analysis(
        self, 
        graph: UserKnowledgeGraph, 
        analysis: Dict[str, Any]
    ) -> None:
        """
        Update knowledge graph with analysis results.
        
        Args:
            graph: UserKnowledgeGraph to update
            analysis: Parsed analysis from Gemini
        """
        # Update quick facts (merge, keep unique, limit to 10)
        new_facts = analysis.get("quick_facts", [])
        existing_facts = set(graph.quick_facts)
        for fact in new_facts:
            if fact not in existing_facts:
                graph.quick_facts.append(fact)
        graph.quick_facts = graph.quick_facts[-10:]  # Keep last 10
        
        # Update interests
        interests = analysis.get("interests", {})
        for category, items in interests.items():
            if not items:
                continue
            
            if category not in graph.interests:
                graph.interests[category] = {}
            
            for name, details in items.items():
                if name in graph.interests[category]:
                    # Update existing interest
                    node = graph.interests[category][name]
                    node.details.update(details)
                    node.mention_count += 1
                    node.last_mentioned = datetime.now()
                else:
                    # Add new interest
                    graph.interests[category][name] = InterestNode(
                        name=name,
                        details=details,
                        last_mentioned=datetime.now(),
                        mention_count=1
                    )
        
        # Update personal info
        personal = analysis.get("personal", {})
        if personal:
            graph.personal.update(personal)
        
        # Update social info
        social = analysis.get("social", {})
        if social:
            # Merge friends lists
            if "friends_mentioned" in social:
                existing_friends = set(graph.social.get("friends_mentioned", []))
                new_friends = set(social.get("friends_mentioned", []))
                graph.social["friends_mentioned"] = list(existing_friends | new_friends)
            
            # Update other social fields
            for key, value in social.items():
                if key != "friends_mentioned" and value:
                    graph.social[key] = value
        
        # Update patterns
        patterns = analysis.get("patterns", {})
        if patterns:
            # Merge typical topics
            if "typical_topics" in patterns:
                existing_topics = set(graph.typical_topics)
                new_topics = set(patterns.get("typical_topics", []))
                graph.typical_topics = list(existing_topics | new_topics)[-10:]
            
            # Merge active hours
            if "active_hours" in patterns:
                existing_hours = set(graph.active_hours)
                new_hours = set(patterns.get("active_hours", []))
                graph.active_hours = sorted(list(existing_hours | new_hours))
        
        graph.updated_at = datetime.now()
    
    async def run_nightly_analysis(
        self, 
        messages_by_user: Dict[int, List[ChatMessage]]
    ) -> Dict[int, UserKnowledgeGraph]:
        """
        Run analysis for all users' messages from the day.
        
        Args:
            messages_by_user: Dictionary mapping user_id to their messages
            
        Returns:
            Dictionary mapping user_id to updated knowledge graphs
        """
        results = {}
        
        logger.info(f"Starting nightly analysis for {len(messages_by_user)} users")
        
        for user_id, messages in messages_by_user.items():
            if not messages:
                continue
            
            username = messages[0].username
            graph = await self.analyze_user_messages(user_id, username, messages)
            
            if graph:
                results[user_id] = graph
        
        logger.info(f"Nightly analysis complete. Updated {len(results)} user profiles.")
        return results


class DailyMessageCollector:
    """
    Collects messages from Firebase for daily analysis.
    """
    
    def __init__(self, firebase_db):
        """
        Initialize collector.
        
        Args:
            firebase_db: Firebase Firestore client
        """
        self._db = firebase_db
    
    def get_yesterday_messages(self) -> Dict[int, List[ChatMessage]]:
        """
        Get all messages from yesterday grouped by user.
        
        Returns:
            Dictionary mapping user_id to list of their messages
        """
        if not self._db:
            logger.warning("Firebase not available")
            return {}
        
        try:
            # Calculate yesterday's date range
            now = datetime.now()
            yesterday_start = datetime(now.year, now.month, now.day) - timedelta(days=1)
            yesterday_end = yesterday_start + timedelta(days=1)
            
            # Query messages from yesterday
            messages_ref = self._db.collection('messages')
            query = messages_ref.where(
                'timestamp', '>=', yesterday_start
            ).where(
                'timestamp', '<', yesterday_end
            ).order_by('timestamp')
            
            docs = query.stream()
            
            # Group by user
            messages_by_user: Dict[int, List[ChatMessage]] = {}
            
            for doc in docs:
                data = doc.to_dict()
                user_id = data.get('user_id')
                
                if user_id:
                    message = ChatMessage(
                        user_id=user_id,
                        username=data.get('username', 'Unknown'),
                        text=data.get('text', ''),
                        message_id=data.get('message_id', 0),
                        timestamp=data.get('timestamp', datetime.now())
                    )
                    
                    if user_id not in messages_by_user:
                        messages_by_user[user_id] = []
                    messages_by_user[user_id].append(message)
            
            logger.info(f"Collected {sum(len(m) for m in messages_by_user.values())} messages from yesterday")
            return messages_by_user
            
        except Exception as e:
            logger.error(f"Error collecting yesterday's messages: {e}")
            return {}
