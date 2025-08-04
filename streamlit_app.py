#!/usr/bin/env python3
"""
Enhanced Email Chat Application with Streamlit
A high-performance app to chat with your emails using Claude AI and Voyage embeddings.

Required packages:
pip install streamlit anthropic voyageai numpy scikit-learn pandas tqdm python-dateutil
"""

import streamlit as st
import mailbox
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Dict, Any, Optional
import pickle
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import threading
import time
import uuid
from email.header import decode_header
from dateutil import parser
import datetime as dt # aliased to avoid conflict with datetime.datetime
# Add this near the top of your file, after the imports
from dateutil import tz

# Set page config as the very first Streamlit command
st.set_page_config(
    page_title="Email Chat Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Timezone mapping to fix dateutil warnings
TIMEZONE_MAPPING = {
    'EDT': -4,  # Eastern Daylight Time
    'EST': -5,  # Eastern Standard Time  
    'PDT': -7,  # Pacific Daylight Time
    'PST': -8,  # Pacific Standard Time
    'UT': 0,    # Universal Time
    'GMT': 0,   # Greenwich Mean Time
    'CST': -6,  # Central Standard Time
    'CDT': -5,  # Central Daylight Time
    'MST': -7,  # Mountain Standard Time
    'MDT': -6,  # Mountain Daylight Time
}


try:
    import anthropic
    import voyageai
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.error("Please install: pip install streamlit anthropic voyageai numpy scikit-learn pandas tqdm python-dateutil")
    st.stop()

class EmailProcessor:
    def __init__(self):
        self.emails = []
        self.embeddings = []
        self.voyage_client = None
        self.claude_client = None
        
    def set_api_keys(self, voyage_key: str, claude_key: str):
        """Set API keys for Voyage and Claude"""
        try:
            self.voyage_client = voyageai.Client(api_key=voyage_key)
            self.claude_client = anthropic.Anthropic(api_key=claude_key)
            return True
        except Exception as e:
            st.error(f"Error setting API keys: {e}")
            return False

    def _decode_header(self, header_value):
        """Properly decode email headers - enhanced to handle all edge cases"""
        if header_value is None:
            return "Unknown"

        # Handle Header objects specifically
        if hasattr(header_value, '__class__') and 'Header' in str(type(header_value)):
            try:
                return str(header_value)
            except Exception:
                return "Unknown"

        try:
            if isinstance(header_value, str):
                return header_value

            decoded_parts = decode_header(str(header_value))
            decoded_string = ""

            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    if encoding:
                        decoded_string += part.decode(encoding, errors='ignore')
                    else:
                        decoded_string += part.decode('utf-8', errors='ignore')
                else:
                    decoded_string += str(part)

            return decoded_string.strip()
        except Exception:
            # Fallback: convert to string
            try:
                return str(header_value)
            except Exception:
                return "Unknown"
    
    def parse_mbox_file(self, file_path: str) -> List[Dict]:
        """Parse MBOX file and extract ALL email data with progress tracking - optimized for large files"""
        emails = []
        try:
            file_size = os.path.getsize(file_path)
            st.info(f"Processing {file_size / (1024**3):.2f} GB MBOX file...")
            
            mbox = mailbox.mbox(file_path)
            
            st.info("Counting emails in MBOX file...")
            total_emails = len(mbox)
            st.success(f"Found {total_emails:,} emails to process")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            email_counter = st.empty()
            
            chunk_size = 1000
            
            for i, message in enumerate(mbox):
                if i % 100 == 0:
                    progress = (i + 1) / total_emails
                    progress_bar.progress(progress)
                    status_text.text(f"Processing email {i+1:,}/{total_emails:,} ({progress*100:.1f}%)")
                    email_counter.text(f"Processed: {len(emails):,} emails")
                
                try:
                    email_data = {
                        'id': str(uuid.uuid4()),
                        'subject': self._decode_header(message.get('Subject')),
                        'from': self._decode_header(message.get('From')),
                        'to': self._decode_header(message.get('To')),
                        'date': self._decode_header(message.get('Date')),
                        'body': self._extract_body(message),
                        'message_id': self._decode_header(message.get('Message-ID')) or f'email_{i}'
                    }
                    
                    email_text = f"Subject: {email_data['subject']}\n"
                    email_text += f"From: {email_data['from']}\n"
                    email_text += f"To: {email_data['to']}\n"
                    email_text += f"Date: {email_data['date']}\n"
                    email_text += f"Body: {email_data['body'][:3000]}"
                    
                    email_data['full_text'] = email_text
                    emails.append(email_data)
                    
                    if len(emails) % chunk_size == 0:
                        self._save_intermediate_progress(emails)
                        
                except Exception as email_error:
                    print(f"Error processing email {i}: {email_error}")
                    continue
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Completed processing {len(emails):,} emails")
            email_counter.empty()
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
                    
        except Exception as e:
            st.error(f"Error parsing MBOX file: {e}")
            if emails:
                st.warning(f"Partial processing completed. Retrieved {len(emails):,} emails before error.")
            return emails
            
        return emails
    
    def _extract_body(self, message) -> str:
        """Extract body text from email message"""
        body = ""
        
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
                    except:
                        continue
        else:
            try:
                body = message.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                body = str(message.get_payload())
        
        body = re.sub(r'<[^>]+>', '', body)
        body = re.sub(r'\s+', ' ', body).strip()
        
        return body[:2000]
    
    def _save_intermediate_progress(self, emails: List[Dict]):
        """Save intermediate progress to prevent data loss"""
        try:
            temp_data = {
                'emails': emails,
                'timestamp': datetime.now().isoformat(),
                'partial': True
            }
            with open('temp_emails_progress.pkl', 'wb') as f:
                pickle.dump(temp_data, f)
        except Exception as e:
            print(f"Warning: Could not save intermediate progress: {e}")
    
    def create_embeddings_batch(self, emails: List[Dict]):
        """Create embeddings for all emails using optimized batching - handles large datasets"""
        if not self.voyage_client:
            raise Exception("Voyage API key not set")
        
        total_emails = len(emails)
        st.info(f"Creating embeddings for {total_emails:,} emails...")
        
        texts = [email['full_text'] for email in emails]
        embeddings = []
        
        if total_emails > 10000:
            batch_size = 50
        elif total_emails > 5000:
            batch_size = 75
        else:
            batch_size = 100
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        embeddings_counter = st.empty()
        
        failed_batches = 0
        
        for i in range(0, len(texts), batch_size):
            batch_num = i // batch_size + 1
            progress = batch_num / total_batches
            
            progress_bar.progress(progress)
            status_text.text(f"Creating embeddings: batch {batch_num}/{total_batches}")
            embeddings_counter.text(f"Embeddings created: {len(embeddings):,}/{total_emails:,}")
            
            batch_texts = texts[i:i + batch_size]
            
            max_retries = 3
            for retry in range(max_retries):
                try:
                    result = self.voyage_client.embed(
                        batch_texts, 
                        model="voyage-3.5", 
                        input_type="document"
                    )
                    embeddings.extend(result.embeddings)
                    break
                    
                except Exception as e:
                    if retry == max_retries - 1:
                        st.warning(f"Failed to create embeddings for batch {batch_num} after {max_retries} attempts: {e}")
                        embeddings.extend([[0.0] * 1024 for _ in batch_texts])
                        failed_batches += 1
                    else:
                        time.sleep(2 ** retry)
            
            if batch_num % 50 == 0:
                self._save_embedding_progress(emails[:i + batch_size], embeddings)
        
        progress_bar.progress(1.0)
        status_text.text(f"✅ Completed creating {len(embeddings):,} embeddings")
        if failed_batches > 0:
            st.warning(f"⚠️ {failed_batches} batches failed - using fallback embeddings")
        
        embeddings_counter.empty()
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        self.emails = emails
        self.embeddings = np.array(embeddings)
    
    def _save_embedding_progress(self, emails: List[Dict], embeddings: List):
        """Save embedding progress to prevent data loss"""
        try:
            temp_data = {
                'emails': emails,
                'embeddings': embeddings,
                'timestamp': datetime.now().isoformat(),
                'partial_embeddings': True
            }
            with open('temp_embeddings_progress.pkl', 'wb') as f:
                pickle.dump(temp_data, f)
        except Exception as e:
            print(f"Warning: Could not save embedding progress: {e}")

    def find_relevant_emails(self, query: str, top_k: int = 25) -> List[Dict]:
        """Find most relevant emails for a query with improved relevance"""
        if not self.voyage_client or len(self.embeddings) == 0:
            return []

        try:
            query_result = self.voyage_client.embed([query], model="voyage-3.5", input_type="query")
            query_embedding = np.array(query_result.embeddings[0])

            similarities = cosine_similarity([query_embedding], self.embeddings)[0]

            top_indices = np.argsort(similarities)[-top_k:][::-1]

            relevant_emails = []
            for idx in top_indices:
                if similarities[idx] > 0.1:
                    email_data = self.emails[idx].copy()
                    email_data['similarity'] = float(similarities[idx])
                    # Add final_score to match the display expectations
                    email_data['final_score'] = float(similarities[idx])
                    relevant_emails.append(email_data)

            return relevant_emails
        except Exception as e:
            st.error(f"Error finding relevant emails: {e}")
            return []
    
    def chat_with_claude(self, query: str, relevant_emails: List[Dict], conversation_history: List[Dict] = None) -> str:
        """Enhanced chat with Claude including conversation context"""
        if not self.claude_client:
            return "Claude API key not set"
        
        context = "Here are the most relevant emails for your query:\n\n"
        for i, email in enumerate(relevant_emails[:15], 1): # Use top 15 emails for context
            context += f"Email {i} (Similarity: {email.get('similarity', 0):.2f}):\n"
            context += f"Subject: {email['subject']}\n"
            context += f"From: {email['from']}\n"
            context += f"Date: {email['date']}\n"
            context += f"Body: {email['body'][:800]}...\n\n"
        
        history_context = ""
        if conversation_history:
            recent_history = conversation_history[-6:]
            for msg in recent_history:
                role = "Human" if msg['role'] == 'user' else "Assistant"
                history_context += f"{role}: {msg['content']}\n"
        
        system_prompt = """You are an AI assistant helping users search and understand their email archive. 
        You have access to the user's emails and can answer questions about them.
        
        Guidelines:
        - Be conversational and helpful
        - Reference specific emails when relevant
        - If emails don't contain enough information, say so clearly
        - Summarize findings when dealing with multiple emails
        - Be specific about dates, senders, and subjects when possible
        - If asked about emails from specific time periods or people, focus on those aspects"""

        messages = [{"role": "user", "content": f"""
{system_prompt}

Previous conversation context:
{history_context}

Email context:
{context}

Current question: {query}

Please provide a helpful response based on the email content and conversation context."""}]

        try:
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1500,
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            return f"Error communicating with Claude: {e}"

# NEW: SmartEmailProcessor with enhanced searching capabilities
class SmartEmailProcessor(EmailProcessor):
    def __init__(self):
        super().__init__()
        self.date_patterns = {
            'last week': 7, 'last month': 30, 'last year': 365,
            'this week': 7, 'this month': 30, 'this year': 365,
            'yesterday': 1, 'today': 0
        }
    
    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """Extract structured information from natural language query"""
        intent = {
            'original_query': query, 'semantic_query': query, 'filters': {},
            'date_range': None, 'sender_filter': None, 'subject_keywords': [],
            'priority_boost': {}, 'query_type': 'general'
        }
        
        query_lower = query.lower()
        
        date_info = self._extract_date_info(query_lower)
        if date_info:
            intent['date_range'] = date_info
            intent['filters']['date'] = date_info
        
        sender_info = self._extract_sender_info(query_lower)
        if sender_info:
            intent['sender_filter'] = sender_info
            intent['filters']['sender'] = sender_info
        
        subject_keywords = self._extract_subject_keywords(query_lower)
        if subject_keywords:
            intent['subject_keywords'] = subject_keywords
            intent['filters']['subject'] = subject_keywords
        
        intent['query_type'] = self._classify_query_type(query_lower)
        intent['priority_boost'] = self._get_priority_boost(query_lower, intent['query_type'])
        intent['semantic_query'] = self._clean_semantic_query(query, intent)
        
        return intent
    
    def _extract_date_info(self, query: str) -> Optional[Dict]:
        """Extract date information from query"""
        now = dt.datetime.now()
        
        for pattern, days in self.date_patterns.items():
            if pattern in query:
                if 'last' in pattern or 'yesterday' in pattern:
                    end_date = now
                    start_date = now - dt.timedelta(days=days)
                elif 'this' in pattern or 'today' in pattern:
                    if days == 0:
                        start_date = now.replace(hour=0, minute=0, second=0)
                        end_date = now
                    else:
                        if 'week' in pattern: start_date = now - dt.timedelta(days=now.weekday())
                        elif 'month' in pattern: start_date = now.replace(day=1)
                        elif 'year' in pattern: start_date = now.replace(month=1, day=1)
                        end_date = now
                return {'start': start_date, 'end': end_date, 'type': pattern}
        
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            year = int(year_match.group(1))
            return {'start': dt.datetime(year, 1, 1), 'end': dt.datetime(year, 12, 31), 'type': f'year_{year}'}
        
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december']
        for i, month in enumerate(months, 1):
            if month in query:
                year = now.year
                if year_match: year = int(year_match.group(1))
                end_day = 28 if i == 2 else 30 if i in [4, 6, 9, 11] else 31
                return {'start': dt.datetime(year, i, 1), 'end': dt.datetime(year, i, end_day), 'type': f'month_{month}'}
        
        return None

    def _extract_sender_info(self, query: str) -> Optional[List[str]]:
        """Extract sender information from query - with debugging"""
        try:
            sender_patterns = [r'from\s+([a-zA-Z\s]+)', r'sent by\s+([a-zA-Z\s]+)', r'by\s+([a-zA-Z@\.\s]+)',
                               r'([a-zA-Z]+@[a-zA-Z\.-]+)']
            senders = []
            for pattern in sender_patterns:
                matches = re.findall(pattern, query)
                print(f"DEBUG: Pattern {pattern} found: {matches}")
                senders.extend(matches)

            cleaned_senders = [s.strip() for s in senders if len(s.strip()) > 2]
            result = list(dict.fromkeys(cleaned_senders)) or None
            print(f"DEBUG: Final sender filters: {result}")
            return result
        except Exception as e:
            print(f"ERROR in _extract_sender_info: {e}")
            return None
    
    def _extract_subject_keywords(self, query: str) -> List[str]:
        """Extract important subject keywords"""
        subject_indicators = ['about', 'regarding', 'subject', 'titled', 'called']
        keywords = []
        for indicator in subject_indicators:
            keywords.extend(re.findall(f'{indicator}\\s+([^,\\.!?]+)', query))
        keywords.extend(re.findall(r'"([^"]+)"', query))
        return [kw.strip() for kw in keywords if len(kw.strip()) > 2]
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query for better search strategy"""
        if any(w in query for w in ['meeting', 'calendar', 'schedule']): return 'meeting'
        if any(w in query for w in ['attachment', 'file', 'document']): return 'attachment'
        if any(w in query for w in ['urgent', 'important', 'asap']): return 'urgent'
        if any(w in query for w in ['project', 'task', 'deadline']): return 'project'
        if any(w in query for w in ['invoice', 'payment', 'bill']): return 'financial'
        if any(w in query for w in ['thank', 'thanks']): return 'gratitude'
        if '?' in query: return 'question'
        return 'general'
    
    def _get_priority_boost(self, query: str, query_type: str) -> Dict[str, float]:
        """Get priority boost factors for different aspects"""
        boost = {}
        type_boosts = {'meeting': {'subject': 1.3, 'body': 1.2}, 'urgent': {'subject': 1.4, 'body': 1.3},
                       'project': {'subject': 1.2, 'body': 1.1}, 'financial': {'subject': 1.3, 'from': 1.2},
                       'attachment': {'body': 1.4}}
        if query_type in type_boosts: boost.update(type_boosts[query_type])
        if any(w in query for w in ['latest', 'recent', 'new']): boost['date'] = 1.2
        return boost
    
    def _clean_semantic_query(self, original_query: str, intent: Dict) -> str:
        """Clean query for better semantic embedding"""
        clean_query = original_query
        filter_words = ['from', 'sent by', 'last week', 'last month', 'this year', 'yesterday']
        for word in filter_words:
            clean_query = re.sub(f'\\b{word}\\b', '', clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', clean_query)
        return re.sub(r'\s+', ' ', clean_query).strip()

    def smart_find_relevant_emails(self, query: str, top_k: int = 25) -> List[Dict]:
        """Enhanced email search with intent understanding - with debugging"""
        if not self.voyage_client or len(self.embeddings) == 0:
            return []

        try:
            print(f"DEBUG: Starting smart search for query: {query}")
            intent = self.extract_query_intent(query)
            print(f"DEBUG: Intent extracted: {intent}")

            semantic_scores = self._get_semantic_scores(intent['semantic_query'])
            print(f"DEBUG: Got semantic scores, shape: {semantic_scores.shape}")

            filtered_emails = self._apply_filters_and_boosts(semantic_scores, intent)
            print(f"DEBUG: Applied filters, got {len(filtered_emails)} emails")

            top_indices = np.argsort([e['final_score'] for e in filtered_emails])[-top_k:][::-1]
            result = [filtered_emails[idx] for idx in top_indices if filtered_emails[idx]['final_score'] > 0.05]
            print(f"DEBUG: Returning {len(result)} final results")
            return result
        except Exception as e:
            print(f"ERROR in smart_find_relevant_emails: {e}")
            import traceback
            traceback.print_exc()
            return self.find_relevant_emails(query, top_k)
    
    def _get_semantic_scores(self, semantic_query: str) -> np.ndarray:
        """Get semantic similarity scores"""
        query_result = self.voyage_client.embed([semantic_query], model="voyage-3.5", input_type="query")
        query_embedding = np.array(query_result.embeddings[0])
        return cosine_similarity([query_embedding], self.embeddings)[0]
    
    def _apply_filters_and_boosts(self, semantic_scores: np.ndarray, intent: Dict) -> List[Dict]:
        """Apply filters and calculate boosted scores"""
        filtered_emails = []
        for idx, base_score in enumerate(semantic_scores):
            email = self.emails[idx]
            if intent['date_range'] and not self._passes_date_filter(email, intent['date_range']): continue
            if intent['sender_filter'] and not self._passes_sender_filter(email, intent['sender_filter']): continue
            
            s_boost = self._calculate_subject_boost(email, intent)
            p_boost = self._calculate_priority_boost(email, intent)
            r_boost = self._calculate_recency_boost(email, intent)
            
            email_data = email.copy()
            email_data['similarity'] = float(base_score)
            email_data['final_score'] = float(base_score * s_boost * p_boost * r_boost)
            email_data['boosts_applied'] = {'subject': s_boost, 'priority': p_boost, 'recency': r_boost}
            filtered_emails.append(email_data)
        
        return filtered_emails

    def _passes_date_filter(self, email: Dict, date_range: Dict) -> bool:
        """Check if email passes date filter - with debugging"""
        try:
            email_date_str = email.get('date', '')
            print(f"DEBUG DATE: Raw date type: {type(email_date_str)}")

            if not email_date_str or email_date_str == 'Unknown':
                return False

            # Make sure it's decoded properly
            email_date_str = self._decode_header(email_date_str)

            # Use timezone mapping to avoid warnings
            email_date = parser.parse(email_date_str, fuzzy=True, tzinfos=TIMEZONE_MAPPING).replace(tzinfo=None)
            return date_range['start'].replace(tzinfo=None) <= email_date <= date_range['end'].replace(tzinfo=None)
        except Exception as e:
            print(f"ERROR in _passes_date_filter: {e}")
            return False

    def _passes_sender_filter(self, email: Dict, sender_filters: List[str]) -> bool:
        """Check if email passes sender filter - with debugging"""
        try:
            email_sender = email.get('from', '')

            # Debug: Print the type and value
            print(f"DEBUG: Original sender type: {type(email_sender)}")
            print(f"DEBUG: Original sender value: {email_sender}")

            # Use the existing _decode_header method to handle Header objects
            email_sender = self._decode_header(email_sender)

            # Debug: Print after decoding
            print(f"DEBUG: After decode_header type: {type(email_sender)}")
            print(f"DEBUG: After decode_header value: {email_sender}")

            # Ensure it's a string and convert to lowercase
            email_sender = str(email_sender).lower()

            return any(sender.lower() in email_sender for sender in sender_filters)
        except Exception as e:
            print(f"ERROR in _passes_sender_filter: {e}")
            print(f"ERROR email data: {email}")
            return False

    def _calculate_subject_boost(self, email: Dict, intent: Dict) -> float:
        """Calculate subject-based boost - with debugging"""
        try:
            if not intent['subject_keywords']:
                return 1.0
            boost = 1.0

            # Debug and safely get subject
            subject_raw = email.get('subject', '')
            print(f"DEBUG SUBJECT: Raw subject type: {type(subject_raw)}")
            subject = self._decode_header(subject_raw).lower()

            for keyword in intent['subject_keywords']:
                if keyword.lower() in subject:
                    boost *= 1.3
            return boost
        except Exception as e:
            print(f"ERROR in _calculate_subject_boost: {e}")
            return 1.0

    def _calculate_priority_boost(self, email: Dict, intent: Dict) -> float:
        """Calculate priority-based boost - with debugging"""
        try:
            boost = 1.0
            p_boosts = intent['priority_boost']

            # Debug and safely get subject
            subject_raw = email.get('subject', '')
            print(f"DEBUG PRIORITY: Raw subject type: {type(subject_raw)}")
            subject = self._decode_header(subject_raw).lower()

            # Debug and safely get body
            body_raw = email.get('body', '')
            print(f"DEBUG PRIORITY: Raw body type: {type(body_raw)}")
            body = self._decode_header(body_raw).lower()

            if 'subject' in p_boosts and any(w in subject for w in ['urgent', 'important']):
                boost *= p_boosts['subject']
            if 'body' in p_boosts and any(w in body for w in ['meeting', 'deadline', 'urgent']):
                boost *= p_boosts['body']
            return boost
        except Exception as e:
            print(f"ERROR in _calculate_priority_boost: {e}")
            print(f"ERROR email data keys: {email.keys()}")
            return 1.0

    def _calculate_recency_boost(self, email: Dict, intent: Dict) -> float:
        """Calculate recency-based boost"""
        try:
            email_date_str = email.get('date', '')
            if not email_date_str or email_date_str == 'Unknown':
                return 1.0
            # Use timezone mapping
            email_date = parser.parse(email_date_str, fuzzy=True, tzinfos=TIMEZONE_MAPPING).replace(tzinfo=None)
            days_old = (dt.datetime.now() - email_date).days
            if intent['query_type'] in ['urgent', 'meeting', 'project']:
                if days_old < 7: return 1.3
                if days_old < 30: return 1.1
            return 1.0
        except:
            return 1.0
    
    def get_search_explanation(self, query: str, intent: Dict, relevant_emails: List[Dict]) -> str:
        """Generate explanation of search strategy"""
        explanation = f"🔍 **Search Strategy for:** '{query}'\n\n"
        explanation += f"**Query Type:** {intent['query_type'].title()}\n"
        if intent['date_range']: explanation += f"**Date Filter:** {intent['date_range']['type']}\n"
        if intent['sender_filter']: explanation += f"**Sender Filter:** {', '.join(intent['sender_filter'])}\n"
        if intent['subject_keywords']: explanation += f"**Subject Keywords:** {', '.join(intent['subject_keywords'])}\n"
        if intent['priority_boost']:
            boosts = ', '.join(f"{k}({v:.1f}x)" for k, v in intent['priority_boost'].items())
            explanation += f"**Priority Boosts:** {boosts}\n"
        explanation += f"\n**Found {len(relevant_emails)} relevant emails**"
        return explanation

def save_processed_data(processor):
    """Save processed emails and embeddings"""
    try:
        data = {
            'emails': processor.emails,
            'embeddings': processor.embeddings.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        with open('processed_emails.pkl', 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.error(f"Error saving processed data: {e}")

def load_processed_data():
    """Load previously processed data"""
    try:
        if os.path.exists('processed_emails.pkl'):
            with open('processed_emails.pkl', 'rb') as f:
                data = pickle.load(f)
            return data['emails'], np.array(data['embeddings'])
    except Exception as e:
        st.error(f"Error loading processed data: {e}")
    return None, None

def save_conversation(conv_id: str, messages: List[Dict]):
    """Save conversation to file"""
    try:
        conversations = load_conversations()
        conversations[conv_id] = messages
        with open('conversations.json', 'w') as f:
            json.dump(conversations, f, indent=2)
    except Exception as e:
        print(f"Error saving conversation: {e}")

def load_conversations() -> Dict:
    """Load saved conversations"""
    try:
        if os.path.exists('conversations.json'):
            with open('conversations.json', 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading conversations: {e}")
    return {}

def cleanup_temp_files():
    """Clean up temporary progress files"""
    for temp_file in ['temp_emails_progress.pkl', 'temp_embeddings_progress.pkl']:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Could not remove temp file {temp_file}: {e}")

def recover_partial_progress(processor):
    """Try to recover from partial processing"""
    # Placeholder for recovery logic, can be expanded
    st.info("Recovery feature can be implemented here.")

def process_large_mbox(file_path: str, processor):
    """Process large MBOX file from local filesystem"""
    try:
        start_time = time.time()
        with st.spinner("Parsing large MBOX file..."):
            emails = processor.parse_mbox_file(file_path)
        if not emails:
            st.error("No emails found in the file")
            return
        with st.spinner("Creating embeddings for all emails..."):
            processor.create_embeddings_batch(emails)
        save_processed_data(processor)
        cleanup_temp_files()
        processing_time = time.time() - start_time
        st.session_state.emails_loaded = True
        st.success(f"✅ Successfully processed {len(emails):,} emails in {processing_time/60:.1f} minutes!")
        st.balloons()
        st.rerun()
    except Exception as e:
        st.error(f"Error processing large MBOX file: {e}")
        recover_partial_progress(processor)

def process_uploaded_mbox(uploaded_file, processor):
    """Process smaller uploaded MBOX file"""
    try:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        with st.spinner("Parsing MBOX file..."):
            emails = processor.parse_mbox_file(temp_path)
        if not emails:
            st.error("No emails found in the file")
            return
        with st.spinner("Creating embeddings..."):
            processor.create_embeddings_batch(emails)
        os.remove(temp_path)
        save_processed_data(processor)
        st.session_state.emails_loaded = True
        st.success(f"✅ Successfully processed {len(emails):,} emails!")
        st.rerun()
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        temp_path = f"temp_{uploaded_file.name}"
        if os.path.exists(temp_path):
            os.remove(temp_path)


def get_date_range_safely(emails):
    """Safely get date range from emails by making all datetimes timezone-naive"""
    valid_dates = []
    for email in emails:
        try:
            date_str = email.get('date', '')
            if date_str and date_str != 'Unknown':
                # Parse the date string with timezone mapping
                dt_obj = parser.parse(date_str, fuzzy=True, tzinfos=TIMEZONE_MAPPING)
                # IMPORTANT: Make it timezone-naive before adding to the list
                valid_dates.append(dt_obj.replace(tzinfo=None))
        except (parser.ParserError, TypeError, ValueError):
            # Catch specific parsing errors and continue
            continue

    if valid_dates:
        # Now all datetimes are naive and can be compared
        return f"Date Range: {min(valid_dates).date()} to {max(valid_dates).date()}"

    return "Date Range: Not available"
    
# Streamlit App
def main():
    # Initialize session state
    if 'processor' not in st.session_state:
        # UPDATED: Use the new SmartEmailProcessor
        st.session_state.processor = SmartEmailProcessor()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'emails_loaded' not in st.session_state:
        st.session_state.emails_loaded = False
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    
    if not st.session_state.emails_loaded:
        emails, embeddings = load_processed_data()
        if emails is not None and embeddings is not None:
            st.session_state.processor.emails = emails
            st.session_state.processor.embeddings = embeddings
            st.session_state.emails_loaded = True
    
    with st.sidebar:
        st.title("⚙️ Setup")
        
        st.subheader("API Keys")
        voyage_key = st.text_input("Voyage API Key", type="password", key="voyage_key")
        claude_key = st.text_input("Claude API Key", type="password", key="claude_key")
        
        if st.button("Set API Keys", type="primary"):
            if voyage_key and claude_key:
                if st.session_state.processor.set_api_keys(voyage_key, claude_key):
                    st.success("✅ API keys set successfully!")
                else:
                    st.error("❌ Failed to set API keys")
            else:
                st.error("Please enter both API keys")
        
        st.divider()
        
        st.subheader("📁 Email Import")
        st.markdown("**For large MBOX files (>200MB):**")
        file_path = st.text_input("Enter full path to your MBOX file:", placeholder="/path/to/your/email.mbox")
        
        if file_path and not st.session_state.emails_loaded:
            if os.path.exists(file_path):
                st.info(f"📊 File size: {os.path.getsize(file_path) / (1024**3):.2f} GB")
                if st.button("Process Large MBOX File", type="primary"):
                    process_large_mbox(file_path, st.session_state.processor)
            else:
                st.error("❌ File not found. Please check the path.")
        
        st.markdown("---")
        
        st.markdown("**For small MBOX files (<200MB):**")
        uploaded_file = st.file_uploader("Upload MBOX file", type=['mbox'], key="mbox_file")
        
        if uploaded_file and not st.session_state.emails_loaded:
            if st.button("Process Uploaded File", type="primary"):
                process_uploaded_mbox(uploaded_file, st.session_state.processor)
        
        if st.session_state.emails_loaded:
            st.subheader("📊 Email Stats")
            num_emails = len(st.session_state.processor.emails)
            st.metric("Total Emails", num_emails)
            if num_emails > 0:
                st.text(get_date_range_safely(st.session_state.processor.emails))
        
        st.divider()
        
        st.subheader("💬 Conversations")
        if st.button("New Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()
        
        conversations = load_conversations()
        if conversations:
            selected_conv = st.selectbox("Load Conversation", options=list(conversations.keys()),
                                       format_func=lambda x: f"{x[:8]}... ({len(conversations[x])} msgs)")
            if st.button("Load Selected"):
                st.session_state.messages = conversations[selected_conv]
                st.rerun()
    
    st.title("📧 Email Chat Assistant")
    
    if not st.session_state.emails_loaded:
        st.info("👆 Please set your API keys and upload an MBOX file to start chatting with your emails!")
        with st.expander("📋 Instructions"):
            st.markdown("""
            ### How to use:
            1. **Get API Keys:** from [Voyage AI](https://voyageai.com) and [Anthropic](https://console.anthropic.com).
            2. **Import Emails:** Use the local path for large files or the uploader for smaller ones.
            3. **Start Chatting:** Ask questions like "What did John say about the project last week?" or "Find the invoice from Acme Corp in January".
            """)
    else:
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if prompt := st.chat_input("Ask about your emails..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # UPDATED: Full chat response logic with Smart Search
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing query and searching emails..."):
                        # Use smart search instead of basic search
                        relevant_emails = st.session_state.processor.smart_find_relevant_emails(prompt, top_k=25)

                        if not relevant_emails:
                            response = "I couldn't find any relevant emails for your query. Please try rephrasing your question."
                        else:
                            # Get search explanation
                            intent = st.session_state.processor.extract_query_intent(prompt)
                            search_explanation = st.session_state.processor.get_search_explanation(prompt, intent, relevant_emails)

                            # Get Claude response with conversation history
                            response = st.session_state.processor.chat_with_claude(
                                prompt,
                                relevant_emails,
                                st.session_state.messages[:-1]  # Exclude current message
                            )

                            # Add search explanation to response
                            response = f"{response}\n\n---\n{search_explanation}"

                        st.markdown(response)

                        # Enhanced relevant emails display
                        if relevant_emails:
                            with st.expander(f"📧 Found {len(relevant_emails)} relevant emails (Smart Search)"):
                                for i, email in enumerate(relevant_emails, 1):
                                    boosts = email.get('boosts_applied', {})
                                    boost_text = ""
                                    active_boosts = [f"{k}:{v:.1f}x" for k, v in boosts.items() if v > 1.0]
                                    if active_boosts:
                                        boost_text = f" 🚀 Boosts: {', '.join(active_boosts)}"

                                    st.markdown(f"""
                                    **Email {i}** (Similarity: {email['similarity']:.3f}, Final Score: {email['final_score']:.3f}){boost_text}
                                    - **Subject:** {email['subject']}
                                    - **From:** {email['from']}
                                    - **Date:** {email['date']}
                                    - **Preview:** {email['body'][:200]}...
                                    """)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                save_conversation(st.session_state.conversation_id, st.session_state.messages)

if __name__ == "__main__":
    main()