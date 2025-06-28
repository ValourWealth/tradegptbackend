from django.shortcuts import render
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
import jwt
from datetime import datetime, timedelta
from django.conf import settings
from rest_framework import status
from .models import ChatSession, ChatMessage
from .utils import get_user_from_token
from django.utils.timezone import now
import uuid
from django.utils import timezone


class TradeGPTUserView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        token = request.GET.get("token")
        if not token:
            return Response({"error": "Token is missing"}, status=400)

        try:
            decoded = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            return Response({
                "user_id": decoded.get("user_id"),
                "username": decoded.get("username"),
                "first_name": decoded.get("first_name"),
                "last_name": decoded.get("last_name"),
                "email": decoded.get("email"),
                "subscription_status": decoded.get("subscription_status"),
                "profile_photo": decoded.get("profile_photo"),
                "phone_number": decoded.get("phone_number"),
                "country": decoded.get("country"),
                "state": decoded.get("state"),
                "is_staff": decoded.get("is_staff"),
                "is_superuser": decoded.get("is_superuser"),
            })
        except jwt.ExpiredSignatureError:
            return Response({"error": "Token expired"}, status=401)
        except jwt.InvalidTokenError:
            return Response({"error": "Invalid token"}, status=401)


class StartChatSessionView(APIView):
    def get(self, request):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        session = ChatSession.objects.create(
            session_id=uuid.uuid4(),
            user_id=user["user_id"],
            username=user["username"],
        )
        return Response({"session_id": session.session_id})
    
    # Replace POST method with this logic:
    def post(self, request):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        # ✅ Check if session for today already exists
        today = timezone.now().date()
        existing_session = ChatSession.objects.filter(
            user_id=user["user_id"],
            created_at__date=today
        ).first()

        if existing_session:
            return Response({"session_id": existing_session.session_id})
        else:
            session = ChatSession.objects.create(
                session_id=uuid.uuid4(),
                user_id=user["user_id"],
                username=user["username"],
            )
            return Response({"session_id": session.session_id})




class MessageListCreateView(APIView):
    def post(self, request, session_id):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        data = request.data
        ChatMessage.objects.create(
            session_id=session_id,
            role=data["role"],
            content=data["content"]
        )
        return Response({"message": "Saved"}, status=201)

    def get(self, request, session_id):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        messages = ChatMessage.objects.filter(session_id=session_id).order_by("timestamp")
        return Response([
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in messages
        ])


class UserChatSessionsView(APIView):
    def get(self, request):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        sessions = ChatSession.objects.filter(user_id=user["user_id"]).order_by("-created_at")
        return Response([
            {"session_id": s.session_id, "created_at": s.created_at}
            for s in sessions
        ])


class DailyMessageLimitView(APIView):
    def get(self, request):
        token = request.GET.get("token")
        user = get_user_from_token(token)

        count = ChatMessage.objects.filter(
            session__user_id=user["user_id"],
            timestamp__date=now().date()
        ).count()

        max_allowed = {
            "free": 3,
            "premium": 5,
            "platinum": 10,
        }.get(user["subscription_status"], 3)

        return Response({"count": count, "max": max_allowed})

# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import AllowAny
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# from .utils import get_user_from_token
# import requests
# import re

# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import AllowAny
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# from .utils import get_user_from_token
# import requests



# # import re
# # import logging

# # from django.http import StreamingHttpResponse
# # from rest_framework.views import APIView
# # from rest_framework.permissions import AllowAny
# # from django.views.decorators.csrf import csrf_exempt
# # from django.utils.decorators import method_decorator
# # from openai import OpenAI
# # import time

# # logger = logging.getLogger(__name__)


# # # def clean_special_chars(text):
# # #     import re

# # #     # Remove markdown styling (bold, italic, code)
# # #     text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)
# # #     text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
# # #     text = re.sub(r'\*(.*?)\*', r'\1', text)
# # #     text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)

# # #     # Convert markdown headers (## Section) → "Section:"
# # #     text = re.sub(r'^#{1,6}\s*(.+)$', r'\1:', text, flags=re.MULTILINE)

# # #     # Remove excessive --- or tables like |...|...|
# # #     text = re.sub(r'^\|.*?\|$', '', text, flags=re.MULTILINE)  # remove table lines
# # #     text = re.sub(r'-{3,}', '\n' + '-'*20 + '\n', text)

# # #     # Normalize spacing and line breaks
# # #     text = re.sub(r'\n{2,}', '\n\n', text)
# # #     text = re.sub(r'\s{2,}', ' ', text)

# # #     return text.strip()
# # def clean_special_chars(text):
# #     import re

# #     # Remove markdown styling
# #     text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)  # bold-italic
# #     text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)       # bold
# #     text = re.sub(r'\*(.*?)\*', r'\1', text)           # italic
# #     text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)   # code

# #     # Replace headings (## Heading) with properly formatted section titles
# #     text = re.sub(r'^#{1,6}\s*(.+)$', r'\n\n### \1\n', text, flags=re.MULTILINE)

# #     # Remove markdown tables and separators
# #     text = re.sub(r'\|.*?\|', '', text)         # remove markdown table rows
# #     text = re.sub(r'-{3,}', '\n' + '-'*20 + '\n', text)  # normalize separators

# #     # Normalize spacing
# #     text = re.sub(r'\n{2,}', '\n\n', text)
# #     text = re.sub(r'\s{2,}', ' ', text)

# #     return text.strip()



# # def normalize_query_type(raw):
# #     raw = raw.lower().strip()
# #     if "price" in raw and "chart" in raw:
# #         return "price_chart"
# #     elif "news" in raw:
# #         return "recent_news"
# #     elif "fundamental" in raw or "technical" in raw:
# #         return "fundamental_analysis"
# #     else:
# #         return "default"

# # @method_decorator(csrf_exempt, name='dispatch')
# # class DeepSeekChatView(APIView):
# #     permission_classes = [AllowAny]

# #     def post(self, request):
# #         try:
# #             data = request.data

# #             symbol = data.get("symbol", "N/A")
# #             name = data.get("name", "N/A")
# #             query_type = normalize_query_type(data.get("queryType", "default"))
# #             price = data.get("price", "N/A")
# #             open_ = data.get("open", "N/A")
# #             high = data.get("high", "N/A")
# #             low = data.get("low", "N/A")
# #             previous_close = data.get("previousClose", "N/A")
# #             volume = data.get("volume", "N/A")
# #             trend = data.get("trend", "N/A")
# #             news_list = data.get("news", [])

# #             news_lines = ""
# #             for item in news_list[:5]:
# #                 headline = item.get("headline", "No headline")
# #                 time_str = item.get("time", "Unknown time")
# #                 category = item.get("category", "General")
# #                 news_lines += f"- {headline} at {time_str} | {category}\n"

# #             if not news_lines.strip():
# #                 news_lines = "No major headlines available."

# #             # Build prompt
# #             if query_type == "price_chart":
# #                 prompt = f"""
# # Act as a financial data analyst. Generate a markdown section showing recent price action for {name} ({symbol}). Include:
# # - Volatility patterns
# # - Trend direction
# # - Notable price movements

# # ## Price Movements  
# # Price: ${price}, Open: ${open_}, High: ${high}, Low: ${low}, Previous Close: ${previous_close}  
# # Volume: {volume}  
# # Trend: {trend}
# # """
# #             elif query_type == "recent_news":
# #                 prompt = f"""
# # Act as a financial news summarizer. Provide a markdown list of the most recent headlines for {name} ({symbol}). Highlight insights by theme.

# # ## Recent News  
# # {news_lines}
# # """
# #             elif query_type == "fundamental_analysis":
# #                 prompt = f"""
# # Act as an expert financial analyst. Provide a markdown breakdown of {name} ({symbol}).

# # ## Company Overview  
# # **Symbol:** {symbol}  
# # **Company:** {name}  
# # **Price:** ${price}  
# # **Open:** ${open_}  
# # **High:** ${high}  
# # **Low:** ${low}  
# # **Previous Close:** ${previous_close}  
# # **Volume:** {volume}  
# # **Trend:** {trend}  

# # ## News Headlines  
# # {news_lines}

# # ## Key Financial Metrics  
# # List valuation ratios, margins, ROE, and KPIs.

# # ## Strategic Initiatives  
# # Mention growth areas or major projects.

# # ## Upcoming Events  
# # Include earnings dates and financial releases.

# # ## Analyst Insights  
# # Summarize bullish/bearish sentiment.

# # ## Risks  
# # Mention major financial or regulatory risks.
# # """
# #             else:
# #                 prompt = f"""
# # Act as a professional trader. Based on recent price and news data, suggest a technical trade idea for {name} ({symbol}) including entry, stop-loss, target, and reasoning.

# # **Symbol:** {symbol}  
# # **Company:** {name}  
# # **Price:** ${price}  
# # **Open:** ${open_}  
# # **High:** ${high}  
# # **Low:** ${low}  
# # **Previous Close:** ${previous_close}  
# # **Volume:** {volume}  
# # **Trend:** {trend}  

# # ## News Headlines  
# # {news_lines}

# # ## Trade Setup  
# # Explain entry, stop-loss, target and technical indicators.
# # """

# #             # Streamed Response
# #             client = OpenAI(
# #                 api_key="sk-fd092005f2f446d78dade7662a13c896",
# #                 base_url="https://api.deepseek.com"
# #             )

# #             response = client.chat.completions.create(
# #                 model="deepseek-chat",
# #                 messages=[
# #                     {"role": "system", "content": "You are TradeGPT, a professional market analyst."},
# #                     {"role": "user", "content": prompt}
# #                 ],
# #                 stream=True
# #             )

# #             def stream():
# #                 for chunk in response:
# #                     content = chunk.choices[0].delta.content
# #                     if content:
# #                         # yield f"data: {content}\n\n"  # Correct SSE format
# #                          # Ensure proper line breaks and spacing
# #                         # content = content.replace("\n", "\n\n").replace("**", "** ")
# #                         content = clean_special_chars(content)

# #                         yield f"data: {content}\n\n"
                        

# #             return StreamingHttpResponse(stream(), content_type="text/event-stream")


# #         except Exception as e:
# #             logger.error(f"Streaming error: {str(e)}")
# #             return Response({"error": str(e)}, status=500)






import re
import logging
import time

from django.http import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from openai import OpenAI

logger = logging.getLogger(__name__)


# def clean_special_chars(text):
#     # Remove markdown styling
#     text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)
#     text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
#     text = re.sub(r'\*(.*?)\*', r'\1', text)
#     text = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', text)
#     # Remove markdown headings like ### 1. Tesla (TSLA) → just "1. Tesla (TSLA)"
#     text = re.sub(r'^#{1,6}\s*(\d+\.\s?[A-Z].+)', r'\1', text, flags=re.MULTILINE)

# # For all other headings, you can still keep optional formatting (or remove this too)
#     text = re.sub(r'^#{1,6}\s*(.+)$', r'\n\n\1\n', text, flags=re.MULTILINE)


#     # Replace headings (## Heading) with properly formatted section titles
#     text = re.sub(r'^#{1,6}\s*(.+)$', r'\n\n### \1\n', text, flags=re.MULTILINE)

#     # Remove markdown tables and separators
#     text = re.sub(r'\|.*?\|', '', text)  # remove markdown table rows
#     text = re.sub(r'-{3,}', '\n' + '-' * 20 + '\n', text)

#     # Normalize spacing
#     text = re.sub(r'\n{2,}', '\n\n', text)
#     text = re.sub(r'\s{2,}', ' ', text)

#     return text.strip()

# def clean_special_chars(text):
#     import html

#     # Decode HTML entities
#     text = html.unescape(text)

#     # Remove unwanted sections
#     text = re.sub(r'<h2>.*?(Response to User|Analysis|Summary|html).*?</h2>', '', text, flags=re.IGNORECASE)

#     # Remove raw section titles
#     text = re.sub(r'\b(Response\s*to\s*User|Analysis|Summary|html)\b', '', text, flags=re.IGNORECASE)

#     # Only preserve safe tags
#     allowed_tags = ["p", "ul", "ol", "li", "b", "br"]
#     tag_pattern = re.compile(r'</?([a-z0-9]+)[^>]*>', flags=re.IGNORECASE)

#     def preserve_tag(match):
#         tag = match.group(1).lower()
#         return match.group(0) if tag in allowed_tags else ''

#     text = tag_pattern.sub(preserve_tag, text)

#     # Remove backticks
#     text = text.replace("```", "")

#     # De-glue stuck words
#     text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
#     text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)

#     # Ensure hyphenated bullet points have space
#     text = re.sub(r'-([^\s])', r'- \1', text)

#     # Normalize whitespace
#     text = re.sub(r'\s{2,}', ' ', text)
#     text = re.sub(r'\n{3,}', '\n\n', text)

#     return text.strip()

def clean_special_chars(text):
    import html
    import re

    # Decode HTML entities
    text = html.unescape(text)

    # Remove unwanted headers and repeated sections
    text = re.sub(r'<h2>.*?(Response to User|Analysis|Summary|html).*?</h2>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Response\s*to\s*User|Analysis|Summary|html)\b', '', text, flags=re.IGNORECASE)

    # Only preserve allowed tags
    allowed_tags = ["p", "ul", "ol", "li", "b", "br"]
    tag_pattern = re.compile(r'</?([a-z0-9]+)[^>]*>', flags=re.IGNORECASE)

    def preserve_tag(match):
        tag = match.group(1).lower()
        return match.group(0) if tag in allowed_tags else ''
    text = tag_pattern.sub(preserve_tag, text)

    # Remove backticks/code
    text = text.replace("```", "")

    # Insert space between lowercase-uppercase, letter-digit, digit-letter
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

    # Space between glued lowercase words using brute force (detect long lowercase runs)
    def split_lowercase_words(m):
        word = m.group(1)
        # Use a simple heuristic: try to break into common word pieces
        chunks = re.findall(r'[a-z]{3,}', word)
        return ' '.join(chunks)

    text = re.sub(r'\b([a-z]{15,})\b', split_lowercase_words, text)

    # Add space after hyphen if stuck (e.g., "-text" to "- text")
    text = re.sub(r'-([^\s])', r'- \1', text)

    # Normalize spacing
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def normalize_query_type(raw):
    raw = raw.lower().strip()
    if "price" in raw and "chart" in raw:
        return "price_chart"
    elif "news" in raw:
        return "recent_news"
    elif "fundamental" in raw or "technical" in raw:
        return "fundamental_analysis"
    else:
        return "default"


@method_decorator(csrf_exempt, name='dispatch')
class DeepSeekChatView(APIView):
    permission_classes = [AllowAny]
    

    def post(self, request):
        try:
            data = request.data

            symbol = data.get("symbol", "N/A")
            name = data.get("name", "N/A")
            query_type = normalize_query_type(data.get("queryType", "default"))
            price = data.get("price", "N/A")
            open_ = data.get("open", "N/A")
            high = data.get("high", "N/A")
            low = data.get("low", "N/A")
            previous_close = data.get("previousClose", "N/A")
            volume = data.get("volume", "N/A")
            trend = data.get("trend", "N/A")
            news_list = data.get("news", [])

            MAX_TOKENS = min(max(int(data.get("tokenLimit", 1500)), 1), 8192)
            news_lines = ""
            for item in news_list[:5]:
                headline = item.get("headline", "No headline")
                time_str = item.get("time", "Unknown time")
                category = item.get("category", "General")
                news_lines += f"- {headline} at {time_str} | {category}\n"

            if not news_lines.strip():
                news_lines = "No major headlines available."

            # Build prompt
            if query_type == "price_chart":
                prompt += """
<p><b>Note:</b> Use clean HTML. Wrap each sentence in <p> or each list item in <li><p>...</p></li>. Avoid combining multiple sentences without punctuation or spacing.</p>
"""

                prompt = f"""
You are TradeGPT, a financial data analyst. Respond in clean HTML format with structure and insights.

<h2>Price Movements for {name} ({symbol})</h2>
<p>
<b>Price:</b> ${price}<br/>
<b>Open:</b> ${open_}<br/>
<b>High:</b> ${high}<br/>
<b>Low:</b> ${low}<br/>
<b>Previous Close:</b> ${previous_close}<br/>
<b>Volume:</b> {volume}<br/>
<b>Trend:</b> {trend}
</p>

<h2>Key Observations</h2>
<ul>
  <li>Discuss volatility patterns</li>
  <li>Identify the trend direction</li>
  <li>Highlight any notable price swings or breakout points</li>
</ul>
"""
            elif query_type == "recent_news":
                prompt = f"""
You are TradeGPT, a financial news summarizer. Present recent headlines for {name} ({symbol}) in structured HTML format.

<h2>Recent News for {name} ({symbol})</h2>
<p>Below are the top headlines:</p>
<ul>
{''.join(f"<li>{item.get('headline', 'No headline')} - <i>{item.get('time', 'Unknown time')}</i> | {item.get('category', 'General')}</li>" for item in news_list[:5]) or "<li>No major headlines available.</li>"}
</ul>

<h2>Insights</h2>
<p>Summarize common themes across these stories — sentiment, market impact, or sector movements.</p>
"""
            elif query_type == "fundamental_analysis":
                prompt = f"""
You are TradeGPT, a senior financial analyst. Provide an HTML-structured analysis of {name} ({symbol}).

<h2>Company Overview</h2>
<p>
<b>Symbol:</b> {symbol}<br/>
<b>Company:</b> {name}<br/>
<b>Price:</b> ${price}<br/>
<b>Open:</b> ${open_}<br/>
<b>High:</b> ${high}<br/>
<b>Low:</b> ${low}<br/>
<b>Previous Close:</b> ${previous_close}<br/>
<b>Volume:</b> {volume}<br/>
<b>Trend:</b> {trend}
</p>

<h2>News Headlines</h2>
<ul>
{''.join(f"<li>{item.get('headline', 'No headline')} - <i>{item.get('time', 'Unknown time')}</i> | {item.get('category', 'General')}</li>" for item in news_list[:5]) or "<li>No major headlines available.</li>"}
</ul>

<h2>Key Financial Metrics</h2>
<ul>
  <li>Valuation ratios (P/E, P/B, P/S)</li>
  <li>Profit margins and ROE</li>
  <li>Liquidity and debt levels</li>
</ul>

<h2>Strategic Initiatives</h2>
<p>Mention active projects, expansion efforts, or innovation strategies.</p>

<h2>Upcoming Events</h2>
<p>Include earnings announcements, dividends, or major releases.</p>

<h2>Analyst Insights</h2>
<p>Summarize sentiment — bullish, bearish, neutral — and why.</p>

<h2>Industry Trends</h2>
<ul>
  <li>Sector performance and headwinds</li>
  <li>Fed policy impact</li>
  <li>Inflation and macroeconomic signals</li>
</ul>

<h2>Buy and Sell Reasons</h2>
<ul>
  <li><b>Buy:</b> Strong earnings, favorable outlook, undervaluation</li>
  <li><b>Sell:</b> Weak guidance, overvaluation, increasing competition</li>
</ul>

<h2>Risks</h2>
<p>Highlight financial, regulatory, or macroeconomic risks.</p>
"""

#             if query_type == "price_chart":
#                 prompt = f"""
# Act as a financial data analyst. Generate a markdown section showing recent price action for {name} ({symbol}). Include:
# - Volatility patterns
# - Trend direction
# - Notable price movements

# ## Price Movements  
# Price: ${price}, Open: ${open_}, High: ${high}, Low: ${low}, Previous Close: ${previous_close}  
# Volume: {volume}  
# Trend: {trend}
# """
#             elif query_type == "recent_news":
#                 prompt = f"""
# Act as a financial news summarizer. Provide a markdown list of the most recent headlines for {name} ({symbol}). Highlight insights by theme.

# ## Recent News  
# {news_lines}
# """
#             elif query_type == "fundamental_analysis":
#                 prompt = f"""
# Act as an expert financial analyst. Provide a detailed markdown breakdown of {name} ({symbol}).

# ## Company Overview  
# **Symbol:** {symbol}  
# **Company:** {name}  
# **Price:** ${price}  
# **Open:** ${open_}  
# **High:** ${high}  
# **Low:** ${low}  
# **Previous Close:** ${previous_close}  
# **Volume:** {volume}  
# **Trend:** {trend}  

# ## News Headlines  
# {news_lines}

# ## Key Financial Metrics  
# List valuation ratios, margins, ROE, and KPIs.

# ## Strategic Initiatives  
# Mention growth areas or major projects.

# ## Upcoming Events  
# Include earnings dates and financial releases.

# ## Analyst Insights  
# Summarize bullish/bearish sentiment.

# ## Industry Trends  
# Discuss broader sector or industry movements that may influence this stock. Include trends such as economic indicators, Fed policies, sector performance, or geopolitical factors. For example:  
# - Technology sector resilience  
# - Fed interest rate outlook  
# - Inflation and consumer demand  
# - Global supply chain effects

# ## Buy and Sell Reasons  
# - **Buy:** List technical and fundamental reasons to enter a long trade now.  
# - **Sell:** List risks such as weakening earnings, competition, valuation concerns, or macro trends.

# ## Risks  
# Mention major financial or regulatory risks.
# """

#             else:
#                     prompt = f"""
# Act as a senior technical analyst and trader. Provide a detailed markdown-based trade ideas valour setup for {name} ({symbol}) based on the latest market data and headlines. Ensure that all sections below are filled with actionable insights.

# **Symbol:** {symbol}  
# **Company:** {name}  
# **Price:** ${price}  
# **Open:** ${open_}  
# **High:** ${high}  
# **Low:** ${low}  
# **Previous Close:** ${previous_close}  
# **Volume:** {volume}  
# **Trend:** {trend}  

# ## News Headlines  
# {news_lines}

# ## Trade Ideas setup by valourGpt  
# Explain entry price, stop-loss, target price, and supporting technical indicators like RSI, MACD, volume trend, support/resistance, or moving averages. Mention any candlestick patterns if relevant.

# ## Key Financial Metrics (Trailing Twelve Months)  
# Include EPS, Gross Margin, Net Margin, Operating Margin, P/E, P/B, P/S, ROA, ROE, Debt/Equity, and Current Ratio. Compare to sector medians if possible.

# ## Upcoming Events  
# Mention scheduled earnings, economic data releases, product launches, or market-moving events that could affect the stock.

# ## Analyst Insights  
# Summarize valuation stance, growth potential, profitability strengths, and momentum. Include any recent earnings revisions or institutional commentary.

# ## Competitors  
# List 2–3 direct competitors. Mention Amazon/Google/Microsoft-type peers and what differentiates this company.

# ## Unique Value Proposition  
# Describe what makes this company valuable long-term — e.g., technology leadership, distribution advantage, IP, customer base, etc.

# ## Buy and Sell Reasons  
# - **Buy:** List technical and fundamental reasons to enter a long trade now.  
# - **Sell:** List risks such as weakening earnings, competition, valuation concerns, or macro trends.
# """
            else:
                prompt = f"""
You are TradeGPT, a senior technical analyst and trader. Respond in clean HTML using headings, paragraphs, and bullet points. Provide a structured breakdown of market data for {name} ({symbol}).

<h2>Stock Snapshot</h2>
<p>
<b>Symbol:</b> {symbol} <br/>
<b>Company:</b> {name} <br/>
<b>Price:</b> ${price} <br/>
<b>Open:</b> ${open_} <br/>
<b>High:</b> ${high} <br/>
<b>Low:</b> ${low} <br/>
<b>Previous Close:</b> ${previous_close} <br/>
<b>Volume:</b> {volume} <br/>
<b>Trend:</b> {trend}
</p>

<h2>News Headlines</h2>
<p>{news_lines.replace("-", "<br/>-")}</p>

<h2>Trade Ideas Setup by ValourGPT</h2>
<ul>
  <li><b>Entry Price:</b> Based on recent support/resistance levels</li>
  <li><b>Stop-Loss:</b> Below key support</li>
  <li><b>Target:</b> Based on trend projections</li>
  <li><b>Indicators:</b> RSI, MACD, candlestick patterns, volume trends</li>
</ul>

<h2>Key Financial Metrics (TTM)</h2>
<ul>
  <li>EPS, P/E, P/B, Net Margin, ROA, ROE, Debt/Equity</li>
  <li>Compare with sector median where possible</li>
</ul>

<h2>Upcoming Events</h2>
<p>List upcoming earnings, releases, or macroeconomic data relevant to the stock.</p>

<h2>Analyst Insights</h2>
<p>Discuss sentiment, earnings revisions, and institutional outlook.</p>

<h2>Competitors</h2>
<p>Mention 2–3 similar companies and what sets this one apart.</p>

<h2>Unique Value Proposition</h2>
<p>Describe core strengths — tech advantage, product ecosystem, IP, etc.</p>

<h2>Buy and Sell Reasons</h2>
<ul>
  <li><b>Buy:</b> Positive trend, financial health, momentum</li>
  <li><b>Sell:</b> Valuation risk, competition, negative news</li>
</ul>
"""


            # Initialize client
            client = OpenAI(
                api_key="sk-fd092005f2f446d78dade7662a13c896",
                base_url="https://api.deepseek.com"
            )

            # Generate response (with token limit)
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are TradeGPT, a professional market analyst."},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                
                max_tokens=MAX_TOKENS
            )

            def stream():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield f"data: {clean_special_chars(content)}\n\n"

            return StreamingHttpResponse(stream(), content_type="text/event-stream")

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            return Response({"error": str(e)}, status=500)


# ===============================================================================
# # direct chat 
@method_decorator(csrf_exempt, name='dispatch')
class DirectChatAIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            message = request.data.get("message", "").strip()
            if not message:
                return Response({"error": "Message is required."}, status=400)
            
            prompt = f"""
You are TradeGPT, a professional financial analyst and assistant. Respond using clean HTML with headings and bullet points.

<h2>Response to User</h2>
<p>{message}</p>

<h2>Analysis</h2>
<ul>
  <li>Break down the query in clear financial terms</li>
  <li>Use real-world examples if relevant</li>
  <li>Highlight any actionable ideas</li>
</ul>

<h2>Summary</h2>
<p>Offer a final conclusion or advice based on the above content.</p>
"""


#             prompt = f"""
# You are TradeGPT, a professional market analyst and assistant. Respond clearly in markdown format and provide complete explanations.

# User: {message}
# """

            client = OpenAI(
                api_key="sk-fd092005f2f446d78dade7662a13c896",
                base_url="https://api.deepseek.com"
            )

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are TradeGPT, a helpful financial assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                max_tokens=1200
            )

            def stream():
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield f"data: {clean_special_chars(content)}\n\n"

            return StreamingHttpResponse(stream(), content_type="text/event-stream")

        except Exception as e:
            logger.error(f"Direct chat error: {str(e)}")
            return Response({"error": str(e)}, status=500)
            