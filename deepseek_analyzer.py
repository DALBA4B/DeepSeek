# deepseek_analyzer.py
"""
DeepSeek-based analyzer for message analysis and knowledge graph building.
Uses DeepSeek API with reasoning for more intelligent analysis.
"""

import logging
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from openai import OpenAI

from models import ChatMessage, InterestStatus
from graph_memory import UserKnowledgeGraph, KnowledgeGraphManager, InterestNode, TopicCategory

logger = logging.getLogger(__name__)


# DeepSeek prompt for analyzing messages and building knowledge graph
DEEPSEEK_ANALYSIS_PROMPT = """Ты извлекаешь ОБЪЕКТИВНЫЕ ФАКТЫ из сообщений пользователя.

Проанализируй сообщения пользователя {username} и выпиши только ПРЯМО СКАЗАННЫЕ факты.

СООБЩЕНИЯ:
{messages}

Верни JSON в ТОЧНО таком формате:
{{
    "facts": ["факт 1 если пользователь прямо это сказал", "факт 2"],
    "interests": {{
        "pets": ["животное 1", "животное 2"],
        "gaming": ["игра 1"],
        "food": ["еда 1"],
        "sports": ["спорт 1"],
        "music": ["жанр 1"],
        "other": ["интерес 1"]
    }}
}}

ВАЖНО - ТОЛЬКО ФАКТЫ:
1. "Я люблю собак" = факт "Любит собак"
2. "Я программист" = факт "Программист"
3. "Я учусь в школе" = факт "Учится в школе"
4. НЕ анализируй эмоции, настроение, стиль общения - это не факты!
5. НЕ писиходелируй - только что ЯВНО сказано в сообщениях
6. Если в интересах нет ничего - оставь объект пустым {{}}
7. Возвращай ТОЛЬКО JSON без пояснений"""


class DeepSeekAnalyzer:
    """Analyzes user messages using DeepSeek API with reasoning."""
    
    def __init__(self, api_key: str, knowledge_manager: Optional[KnowledgeGraphManager] = None):
        """
        Initialize DeepSeek analyzer.
        
        Args:
            api_key: DeepSeek API key
            knowledge_manager: Optional KnowledgeGraphManager instance
        """
        self._api_key = api_key
        self._knowledge_manager = knowledge_manager
        
        try:
            self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            self._model = 'deepseek-reasoner'  # Using reasoner model for intelligent analysis
            logger.info("DeepSeekAnalyzer initialized with deepseek-reasoner model")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSeek: {e}")
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
            messages: List of all messages from the day (including bot responses for context)
            
        Returns:
            Updated UserKnowledgeGraph or None on error
        """
        # Filter out bot's own messages (user_id == -1) from analysis
        # But keep them in context for understanding conversation flow
        user_messages = [msg for msg in messages if msg.user_id != -1 and msg.user_id == user_id]
        
        if not user_messages:
            logger.debug(f"No user messages to analyze for user {user_id}")
            return None
        
        # Format all messages (user + bot) for context, but only analyze user messages
        messages_text = "\n".join([
            f"[{msg.timestamp.strftime('%H:%M')}] {msg.username}: {msg.text}"
            for msg in messages
        ])
        
        prompt = DEEPSEEK_ANALYSIS_PROMPT.format(
            username=username,
            messages=messages_text
        )
        
        try:
            # Call DeepSeek API with reasoning using direct HTTP
            # (OpenAI SDK doesn't support DeepSeek's thinking parameter)
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self._model,
                "max_tokens": 16000,
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 10000
                },
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code != 200:
                raise Exception(f"DeepSeek API error {response.status_code}: {response.text}")
            
            result = response.json()
            
            # Extract the response text (skip thinking)
            response_text = result["choices"][0]["message"]["content"].strip()
            
            # Clean up response (remove markdown if present)
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            response_text = response_text.strip()
            
            # Parse JSON response
            analysis = json.loads(response_text)
            
            # Load or create knowledge graph
            if self._knowledge_manager:
                graph = self._knowledge_manager.get_user_graph(user_id, username)
            else:
                graph = UserKnowledgeGraph(user_id=user_id, username=username)
            
            # Update graph with analysis results and track new facts
            new_facts = self._update_graph_from_analysis(graph, analysis)
            
            logger.info(f"Analysis for {username}: {len(new_facts)} new facts added")
            
            # Attach new facts to graph for display
            graph.new_facts = new_facts
            
            # Save to Firebase if available
            if self._knowledge_manager:
                self._knowledge_manager.save_user_graph(graph)
            
            logger.info(f"Successfully analyzed {len(user_messages)} messages for user {user_id}")
            return graph
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse DeepSeek response as JSON: {e}")
            logger.debug(f"Response was: {response_text[:500]}")
            return None
        except Exception as e:
            logger.error(f"Error analyzing messages for user {user_id}: {e}")
            return None
    
    def _update_graph_from_analysis(
        self, 
        graph: UserKnowledgeGraph, 
        analysis: Dict[str, Any]
    ) -> list:
        """
        Update knowledge graph with analysis results.
        Now handles InterestEntry with status tracking.
        
        Args:
            graph: UserKnowledgeGraph to update
            analysis: Parsed analysis from DeepSeek
            
        Returns:
            List of newly added facts
        """
        # Update facts
        new_facts = analysis.get("facts", [])
        existing_facts = set(graph.quick_facts)
        added_facts_list = []
        
        for fact in new_facts:
            if fact not in existing_facts:
                graph.quick_facts.append(fact)
                added_facts_list.append(fact)
        
        graph.quick_facts = graph.quick_facts[-10:]  # Keep last 10
        
        # Update interests using new InterestEntry structure
        interests = analysis.get("interests", {})
        
        for category_str, items in interests.items():
            if not items:
                continue
            
            # Convert string category to TopicCategory enum
            try:
                category = TopicCategory(category_str)
            except ValueError:
                logger.warning(f"Unknown category: {category_str}")
                continue
            
            for name in items:
                # Use graph's add_interest() for proper versioning
                graph.add_interest(
                    category=category,
                    name=name,
                    status=InterestStatus.LIKES  # Default to likes when mentioned
                )
        
        # Social info is NOT tracked - bot cannot reliably understand social nuances
        # Personal info is NOT tracked - only objective facts and interests matter
        # Patterns are NOT tracked - not important for current use case
        
        graph.updated_at = datetime.now()
        
        # Return list of new facts for display in Telegram
        return added_facts_list


class DailyMessageCollector:
    """Collects messages for daily analysis for nightly DeepSeek analysis."""
    
    def __init__(self, firebase_db=None, memory=None):
        """
        Initialize collector.
        
        Args:
            firebase_db: Optional Firebase client
            memory: Optional Memory instance for fallback
        """
        self.db = firebase_db
        self.memory = memory
    
    async def get_messages_for_day(self, date: Optional[datetime] = None) -> Dict[int, List[ChatMessage]]:
        """
        Get all messages for a specific day grouped by user.
        
        Args:
            date: Date to fetch (defaults to today)
            
        Returns:
            Dict mapping user_id to list of messages
        """
        if date is None:
            date = datetime.now()
        
        messages_by_user = {}
        
        # Try Firebase first if available
        if self.db:
            try:
                docs = self.db.collection("messages").where(
                    "date", "==", date.date().isoformat()
                ).stream()
                
                for doc in docs:
                    msg_data = doc.to_dict()
                    
                    # Parse timestamp from Firebase (stored as string)
                    if isinstance(msg_data.get('timestamp'), str):
                        msg_data['timestamp'] = datetime.fromisoformat(msg_data['timestamp'])
                    
                    msg = ChatMessage(**msg_data)
                    if msg.user_id not in messages_by_user:
                        messages_by_user[msg.user_id] = []
                    messages_by_user[msg.user_id].append(msg)
                
                if messages_by_user:
                    logger.info(f"Loaded {sum(len(m) for m in messages_by_user.values())} messages from Firebase")
                    return messages_by_user
            except Exception as e:
                logger.warning(f"Failed to load messages from Firebase: {e}")
        
        # Fallback to memory
        if self.memory:
            daily_log = self.memory.get_daily_log()
            for msg in daily_log:
                if msg.user_id not in messages_by_user:
                    messages_by_user[msg.user_id] = []
                messages_by_user[msg.user_id].append(msg)
            
            if messages_by_user:
                logger.info(f"Loaded {sum(len(m) for m in messages_by_user.values())} messages from memory")
            
            return messages_by_user
        
        logger.warning("No data source available for daily analysis")
        return {}
    
    async def get_yesterday_messages(self) -> Dict[int, List[ChatMessage]]:
        """
        Get all messages from yesterday grouped by user.
        Convenience method that calls get_messages_for_day() with yesterday's date.
        
        Returns:
            Dict mapping user_id to list of messages from yesterday
        """
        yesterday = datetime.now() - timedelta(days=1)
        return await self.get_messages_for_day(yesterday)
