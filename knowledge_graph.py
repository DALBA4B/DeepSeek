# knowledge_graph.py
"""
Knowledge Graph system for personalized bot responses.
Stores user profiles with interests, patterns, and quick facts.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum

logger = logging.getLogger(__name__)


class TopicCategory(Enum):
    """Categories for topic classification."""
    GAMING = "gaming"
    FOOD = "food"
    EDUCATION = "education"
    WORK = "work"
    ENTERTAINMENT = "entertainment"
    SOCIAL = "social"
    TECH = "tech"
    SPORTS = "sports"
    MUSIC = "music"
    TRAVEL = "travel"
    GENERAL = "general"


# Topic keywords for filtering relevant graph sections
TOPIC_KEYWORDS: Dict[TopicCategory, List[str]] = {
    TopicCategory.GAMING: [
        'игра', 'игры', 'играть', 'играем', 'поиграем', 'катка', 'каток',
        'дота', 'dota', 'дотка', 'доту', 'доте',
        'кс', 'cs', 'csgo', 'cs2', 'контра',
        'рейтинг', 'ранг', 'ммр', 'mmr', 'рейт',
        'стим', 'steam', 'геймер', 'гейминг',
        'лол', 'lol', 'лига', 'легенд',
        'валорант', 'valorant', 'вало',
        'пубг', 'pubg', 'апекс', 'apex',
        'майнкрафт', 'minecraft', 'майн',
        'фортнайт', 'fortnite',
        'консоль', 'плойка', 'ps5', 'xbox',
        'керри', 'саппорт', 'мид', 'хард', 'офлейн',
    ],
    TopicCategory.FOOD: [
        'еда', 'есть', 'поесть', 'кушать', 'жрать',
        'пицца', 'пиццу', 'пиццерия',
        'суши', 'роллы', 'японская',
        'бургер', 'макдак', 'мак', 'kfc',
        'заказ', 'заказать', 'доставка', 'доставку',
        'голод', 'голоден', 'голодный', 'жрать',
        'перекус', 'перекусить', 'снэк',
        'завтрак', 'обед', 'ужин',
        'кофе', 'чай', 'напиток',
        'ресторан', 'кафе', 'столовая',
        'готовить', 'рецепт', 'блюдо',
    ],
    TopicCategory.EDUCATION: [
        'учёба', 'учеба', 'учить', 'учиться',
        'экзамен', 'экзамены', 'зачёт', 'зачет',
        'универ', 'университет', 'вуз', 'институт',
        'школа', 'колледж', 'техникум',
        'лекция', 'лекции', 'пара', 'пары',
        'препод', 'преподаватель', 'учитель',
        'диплом', 'курсовая', 'курсач',
        'сессия', 'семестр', 'каникулы',
        'домашка', 'дз', 'задание',
        'оценка', 'балл', 'рейтинг',
    ],
    TopicCategory.WORK: [
        'работа', 'работать', 'работу',
        'офис', 'офисе', 'удалёнка', 'удаленка',
        'зарплата', 'зп', 'деньги', 'бабки',
        'босс', 'начальник', 'директор',
        'коллега', 'коллеги', 'команда',
        'проект', 'дедлайн', 'задача',
        'митинг', 'созвон', 'звонок',
        'отпуск', 'выходной', 'больничный',
        'карьера', 'повышение', 'увольнение',
    ],
    TopicCategory.ENTERTAINMENT: [
        'фильм', 'кино', 'сериал', 'мультик',
        'смотреть', 'посмотреть', 'глянуть',
        'нетфликс', 'netflix', 'ютуб', 'youtube',
        'аниме', 'анимэ', 'манга',
        'книга', 'читать', 'почитать',
        'концерт', 'театр', 'выставка',
        'клуб', 'бар', 'тусовка', 'вечеринка',
    ],
    TopicCategory.SOCIAL: [
        'встреча', 'встретиться', 'увидеться',
        'друг', 'друзья', 'подруга', 'товарищ',
        'девушка', 'парень', 'отношения',
        'семья', 'родители', 'мама', 'папа',
        'день рождения', 'др', 'праздник',
        'свадьба', 'вечеринка', 'тусовка',
    ],
    TopicCategory.TECH: [
        'комп', 'компьютер', 'ноут', 'ноутбук',
        'телефон', 'смартфон', 'айфон', 'iphone', 'андроид',
        'программа', 'приложение', 'апп', 'app',
        'код', 'кодить', 'программировать',
        'баг', 'ошибка', 'фикс',
        'интернет', 'вайфай', 'wifi',
        'обновление', 'апдейт', 'update',
    ],
    TopicCategory.SPORTS: [
        'спорт', 'тренировка', 'трениться',
        'зал', 'качалка', 'фитнес',
        'футбол', 'баскетбол', 'волейбол',
        'бег', 'бегать', 'пробежка',
        'матч', 'игра', 'чемпионат',
    ],
    TopicCategory.MUSIC: [
        'музыка', 'песня', 'трек', 'альбом',
        'слушать', 'послушать',
        'концерт', 'выступление',
        'группа', 'исполнитель', 'артист',
        'спотифай', 'spotify', 'яндекс музыка',
    ],
    TopicCategory.TRAVEL: [
        'путешествие', 'поездка', 'отпуск',
        'самолёт', 'поезд', 'машина',
        'отель', 'гостиница', 'хостел',
        'виза', 'паспорт', 'билет',
        'страна', 'город', 'море', 'горы',
    ],
}


@dataclass
class InterestNode:
    """A node in the interest graph."""
    name: str
    details: Dict[str, Any] = field(default_factory=dict)
    last_mentioned: datetime = field(default_factory=datetime.now)
    mention_count: int = 1


@dataclass
class UserKnowledgeGraph:
    """
    Knowledge graph for a single user.
    Stores interests, personal info, social connections, and patterns.
    """
    user_id: int
    username: str
    
    # Quick facts for fast context
    quick_facts: List[str] = field(default_factory=list)
    
    # Hierarchical knowledge graph
    interests: Dict[str, Dict[str, InterestNode]] = field(default_factory=dict)
    personal: Dict[str, Any] = field(default_factory=dict)
    social: Dict[str, Any] = field(default_factory=dict)
    
    # Behavioral patterns
    active_hours: List[int] = field(default_factory=list)
    typical_topics: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firebase storage."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "quick_facts": self.quick_facts,
            "knowledge_graph": {
                "interests": {
                    category: {
                        name: {
                            "details": node.details,
                            "last_mentioned": node.last_mentioned.isoformat(),
                            "mention_count": node.mention_count
                        }
                        for name, node in nodes.items()
                    }
                    for category, nodes in self.interests.items()
                },
                "personal": self.personal,
                "social": self.social,
            },
            "patterns": {
                "active_hours": self.active_hours,
                "typical_topics": self.typical_topics,
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserKnowledgeGraph':
        """Create from Firebase document."""
        graph = cls(
            user_id=data.get("user_id", 0),
            username=data.get("username", "Unknown"),
        )
        
        graph.quick_facts = data.get("quick_facts", [])
        
        # Parse knowledge graph
        kg = data.get("knowledge_graph", {})
        
        # Parse interests
        interests_data = kg.get("interests", {})
        for category, nodes in interests_data.items():
            graph.interests[category] = {}
            for name, node_data in nodes.items():
                graph.interests[category][name] = InterestNode(
                    name=name,
                    details=node_data.get("details", {}),
                    last_mentioned=datetime.fromisoformat(
                        node_data.get("last_mentioned", datetime.now().isoformat())
                    ),
                    mention_count=node_data.get("mention_count", 1)
                )
        
        graph.personal = kg.get("personal", {})
        graph.social = kg.get("social", {})
        
        # Parse patterns
        patterns = data.get("patterns", {})
        graph.active_hours = patterns.get("active_hours", [])
        graph.typical_topics = patterns.get("typical_topics", [])
        
        # Parse timestamps
        if "created_at" in data:
            graph.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            graph.updated_at = datetime.fromisoformat(data["updated_at"])
        
        return graph

    def get_relevant_context(self, topics: Set[TopicCategory]) -> str:
        """
        Get relevant parts of the graph based on detected topics.
        
        Args:
            topics: Set of detected topic categories
            
        Returns:
            Formatted context string for DeepSeek
        """
        context_parts = []
        
        # Always include quick facts
        if self.quick_facts:
            context_parts.append(f"Факты о {self.username}: " + ", ".join(self.quick_facts[:5]))
        
        # Add relevant interests
        for topic in topics:
            category = topic.value
            if category in self.interests:
                interests_str = []
                for name, node in self.interests[category].items():
                    if node.details:
                        details_str = ", ".join(f"{k}: {v}" for k, v in node.details.items())
                        interests_str.append(f"{name} ({details_str})")
                    else:
                        interests_str.append(name)
                
                if interests_str:
                    context_parts.append(f"{self.username} интересуется ({category}): " + ", ".join(interests_str))
        
        # Add typical topics if relevant
        if self.typical_topics and TopicCategory.GENERAL in topics:
            context_parts.append(f"Обычные темы {self.username}: " + ", ".join(self.typical_topics[:3]))
        
        return "\n".join(context_parts) if context_parts else ""


class TopicDetector:
    """Detects topics from message text."""
    
    @staticmethod
    def detect_topics(text: str) -> Set[TopicCategory]:
        """
        Detect topic categories from message text.
        
        Args:
            text: Message text to analyze
            
        Returns:
            Set of detected topic categories
        """
        text_lower = text.lower()
        detected = set()
        
        for category, keywords in TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected.add(category)
                    break  # One match per category is enough
        
        # If no specific topic detected, use GENERAL
        if not detected:
            detected.add(TopicCategory.GENERAL)
        
        return detected


class KnowledgeGraphManager:
    """
    Manages knowledge graphs for all users.
    Handles loading, saving, and filtering.
    """
    
    def __init__(self, firebase_db=None):
        """
        Initialize the knowledge graph manager.
        
        Args:
            firebase_db: Firebase Firestore client
        """
        self._db = firebase_db
        self._cache: Dict[int, UserKnowledgeGraph] = {}
        self._topic_detector = TopicDetector()
        logger.info("KnowledgeGraphManager initialized")
    
    def get_user_graph(self, user_id: int, username: str = "Unknown") -> UserKnowledgeGraph:
        """
        Get or create knowledge graph for a user.
        
        Args:
            user_id: Telegram user ID
            username: Username for new graphs
            
        Returns:
            UserKnowledgeGraph instance
        """
        # Check cache first
        if user_id in self._cache:
            return self._cache[user_id]
        
        # Try to load from Firebase
        if self._db:
            try:
                doc = self._db.collection('knowledge_graphs').document(str(user_id)).get()
                if doc.exists:
                    graph = UserKnowledgeGraph.from_dict(doc.to_dict())
                    self._cache[user_id] = graph
                    logger.info(f"Loaded knowledge graph for user {user_id}")
                    return graph
            except Exception as e:
                logger.error(f"Error loading knowledge graph: {e}")
        
        # Create new graph
        graph = UserKnowledgeGraph(user_id=user_id, username=username)
        self._cache[user_id] = graph
        logger.info(f"Created new knowledge graph for user {user_id}")
        return graph
    
    def save_user_graph(self, graph: UserKnowledgeGraph) -> bool:
        """
        Save knowledge graph to Firebase.
        
        Args:
            graph: UserKnowledgeGraph to save
            
        Returns:
            True if saved successfully
        """
        if not self._db:
            logger.warning("Firebase not available, graph not saved")
            return False
        
        try:
            graph.updated_at = datetime.now()
            self._db.collection('knowledge_graphs').document(str(graph.user_id)).set(
                graph.to_dict()
            )
            logger.info(f"Saved knowledge graph for user {graph.user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
            return False
    
    def get_relevant_context_for_message(
        self, 
        user_id: int, 
        message_text: str,
        username: str = "Unknown"
    ) -> str:
        """
        Get relevant context from user's knowledge graph based on message.
        
        Args:
            user_id: Telegram user ID
            message_text: Current message text
            username: Username for new graphs
            
        Returns:
            Formatted context string
        """
        # Detect topics in message
        topics = self._topic_detector.detect_topics(message_text)
        logger.debug(f"Detected topics: {[t.value for t in topics]}")
        
        # Get user's graph
        graph = self.get_user_graph(user_id, username)
        
        # Get relevant context
        return graph.get_relevant_context(topics)
    
    def get_all_cached_graphs(self) -> List[UserKnowledgeGraph]:
        """Get all cached knowledge graphs."""
        return list(self._cache.values())
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
        logger.info("Knowledge graph cache cleared")
