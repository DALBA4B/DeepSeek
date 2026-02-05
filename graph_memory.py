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

from models import InterestEntry, InterestStatus

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
    Stores interests with versioned history, personal info, and patterns.
    """
    user_id: int
    username: str
    
    # Quick facts for fast context
    quick_facts: List[str] = field(default_factory=list)
    
    # Interests: category -> List of InterestEntry (not dict, for versioning)
    interests: Dict[str, List[InterestEntry]] = field(default_factory=dict)
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
            "knowledge_graph": {
                "quick_facts": self.quick_facts,
                "interests": {
                    category: [entry.to_dict() for entry in entries]
                    for category, entries in self.interests.items()
                },
                "personal": self.personal,
                "social": self.social,
                "patterns": {
                    "active_hours": self.active_hours,
                    "typical_topics": self.typical_topics,
                },
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
        
        # Parse knowledge graph (all data is now nested inside)
        kg = data.get("knowledge_graph", {})
        graph.quick_facts = kg.get("quick_facts", [])
        
        # Parse interests (now as list of InterestEntry)
        interests_data = kg.get("interests", {})
        for category, entries_list in interests_data.items():
            graph.interests[category] = []
            for entry_data in entries_list:
                entry = InterestEntry(
                    name=entry_data.get("name", ""),
                    status=InterestStatus(entry_data.get("status", "likes")),
                    added_at=datetime.fromisoformat(
                        entry_data.get("added_at", datetime.now().isoformat())
                    ),
                    current=entry_data.get("current", True)
                )
                graph.interests[category].append(entry)
        
        graph.personal = kg.get("personal", {})
        graph.social = kg.get("social", {})
        
        # Parse patterns (now inside knowledge_graph)
        patterns = kg.get("patterns", {})
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
        
        # Add relevant interests (only current=True entries)
        for topic in topics:
            category = topic.value
            if category in self.interests:
                current_interests = [e for e in self.interests[category] if e.current]
                if current_interests:
                    interests_str = []
                    for entry in current_interests:
                        status_text = "нравится" if entry.status == InterestStatus.LIKES else "не нравится"
                        interests_str.append(f"{entry.name} ({status_text})")
                    
                    if interests_str:
                        context_parts.append(f"{self.username} ({category}): " + ", ".join(interests_str))
        
        # Add typical topics if relevant
        if self.typical_topics and TopicCategory.GENERAL in topics:
            context_parts.append(f"Обычные темы {self.username}: " + ", ".join(self.typical_topics[:3]))
        
        return "\n".join(context_parts) if context_parts else ""
    
    def add_interest(self, category: TopicCategory, name: str, status: InterestStatus) -> None:
        """
        Add or update an interest.
        If interest already exists with different status, mark old as current=False and add new.
        
        Args:
            category: Interest category
            name: Interest name
            status: Interest status (likes/dislikes)
        """
        cat_str = category.value
        
        if cat_str not in self.interests:
            self.interests[cat_str] = []
        
        # Check if interest already exists
        existing_entry = None
        for entry in self.interests[cat_str]:
            if entry.name.lower() == name.lower() and entry.current:
                existing_entry = entry
                break
        
        if existing_entry and existing_entry.status == status:
            # Same status, just update timestamp
            existing_entry.added_at = datetime.now()
            logger.debug(f"Updated existing interest: {name} ({cat_str})")
        elif existing_entry:
            # Status changed - mark old as not current, add new entry
            existing_entry.current = False
            new_entry = InterestEntry(name=name, status=status, current=True)
            self.interests[cat_str].append(new_entry)
            logger.info(f"Interest changed: {name} ({cat_str}) - {existing_entry.status.value} → {status.value}")
        else:
            # New interest
            new_entry = InterestEntry(name=name, status=status, current=True)
            self.interests[cat_str].append(new_entry)
            logger.info(f"Added new interest: {name} ({cat_str}) - {status.value}")
    
    def get_interests(self, category: Optional[TopicCategory] = None) -> Dict[str, List[InterestEntry]]:
        """
        Get current interests, optionally filtered by category.
        
        Args:
            category: Optional category filter
            
        Returns:
            Dictionary of category -> list of InterestEntry (current=True only)
        """
        if category:
            cat_str = category.value
            current_entries = [e for e in self.interests.get(cat_str, []) if e.current]
            return {cat_str: current_entries} if current_entries else {}
        
        # Return all current entries
        result = {}
        for cat, entries in self.interests.items():
            current = [e for e in entries if e.current]
            if current:
                result[cat] = current
        return result
    
    def get_interest_history(self, name: str) -> List[InterestEntry]:
        """
        Get complete history of an interest (all versions).
        
        Args:
            name: Interest name
            
        Returns:
            List of all versions of this interest
        """
        history = []
        for entries_list in self.interests.values():
            for entry in entries_list:
                if entry.name.lower() == name.lower():
                    history.append(entry)
        return sorted(history, key=lambda x: x.added_at)


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
                    data = doc.to_dict()
                    # Validate data is dict, not corrupted
                    if isinstance(data, dict):
                        graph = UserKnowledgeGraph.from_dict(data)
                        self._cache[user_id] = graph
                        logger.info(f"Loaded knowledge graph for user {user_id}")
                        return graph
                    else:
                        logger.warning(f"Corrupted knowledge graph data for user {user_id}, creating new")
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
