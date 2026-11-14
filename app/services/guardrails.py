"""
Pre-LLM guardrails — refuse off-topic questions and jailbreak attempts before they reach (and cost) the LLM.

Pure pattern matching, no model calls. Covers English and Bengali. The order of checks matters:
  1. jailbreak attempts                         → block
  2. clear misuse (code, recipes, essays, …)    → block, even if a property word appears
  3. any real-estate signal                     → allow
  4. general-knowledge / entertainment topics   → block
  5. greetings, thanks, "what can you do"       → allow
  6. short follow-up inside an ongoing chat     → allow (must refer back: "it", "that one", "what about…")
  7. anything else                              → block
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

# ── Patterns ──────────────────────────────────────────────────────────────────

def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


_JAILBREAK = _rx(
    r"\b(ignore|disregard|forget|override|bypass|skip)\b.{0,40}\b(previous|prior|above|earlier|all|your|the|any)\b.{0,25}\b(instructions?|rules?|prompts?|guidelines?|directives?|restrictions?)",
    r"\b(show|reveal|print|repeat|display|output|tell me|give me|what (is|are)|leak)\b.{0,30}\b(system|initial|hidden|original|developer)\s+(prompt|message|instructions?)",
    r"\b(show|reveal|print|repeat|tell me|what are)\b.{0,15}\byour\s+(instructions|rules|prompt|guidelines)\b",
    r"\bsystem\s+prompt\b",
    r"\byou\s+are\s+now\b",
    r"\bfrom\s+now\s+on,?\s+you\b",
    r"\bdo\s+anything\s+now\b",
    r"\b(dan|developer|god|jailbreak|unrestricted|unfiltered)\s+mode\b",
    r"\bjail\s*break",
    r"\b(pretend|act|behave|role-?play)\b.{0,25}\b(no|without)\s+(rules|restrictions|limits|filters|guidelines)",
    r"\b(pretend|act|roleplay|role-play)\s+(to\s+be|as|like)\b.{0,30}\b(dan|ai without|unrestricted|unfiltered|evil|hacker)",
    r"\bnew\s+(instructions|rules)\s*:",
    r"^\s*(system|assistant)\s*:",
    r"<\s*/?\s*(system|instructions?)\s*>",
    # Bengali: "ignore/forget previous instructions", "system prompt", "show your instructions"
    r"(আগের|পূর্বের|সব|সকল)\s*(নির্দেশ|নির্দেশনা|নিয়ম).{0,20}(উপেক্ষা|ভুলে|বাদ)",
    r"সিস্টেম\s*প্রম্পট",
    r"(তোমার|আপনার)\s*(নির্দেশনা|নির্দেশ|প্রম্পট).{0,15}(দেখাও|দেখান|বলো|বলুন)",
)
# "DAN" is only a jailbreak name in capitals ("Dan" is also a person's name)
_JAILBREAK_CASED = re.compile(r"\bDAN\b")

_STRONG_OFF_TOPIC = _rx(
    r"\b(write|create|generate|give me|show me)\b.{0,30}\b(code|function|script|program|class|algorithm|query|regex|snippet)\b",
    r"\b(python|javascript|typescript|java|c\+\+|c#|golang|rust|php|sql|html|css|react|django|flask)\b.{0,30}\b(code|function|script|program|class|loop|error|bug)\b",
    r"\b(code|function|script|program)\b.{0,15}\bin\s+(python|javascript|java|c\+\+|sql|php)\b",
    r"\b(coding|programming|debug|compile|leetcode|stack ?trace)\b",
    r"\brecipes?\b", r"\bhow\s+(do\s+i|to)\s+(cook|bake|make\s+(a\s+)?(cake|pizza|pasta|bread|curry|biryani))\b",
    r"\b(write|compose|create)\b.{0,20}\b(poem|poetry|essay|story|song|lyrics|haiku|limerick|novel|speech|cover letter)\b",
    r"\btranslate\b",
    r"\b(solve|calculate|differentiate|integrate)\b.{0,20}\b(equation|integral|derivative|x\s*[=+\-*/^])",
    r"\b(homework|assignment)\b",
    # Bengali: recipe/cooking, code/program, write a poem/essay/story
    r"রেসিপি|রান্না",
    r"কোড\s*(লিখ|লেখ|দাও|দিন)|প্রোগ্রামিং|পাইথন|জাভাস্ক্রিপ্ট",
    r"(কবিতা|রচনা|গল্প|গান).{0,15}(লেখ|লিখ)",
)

_WEAK_OFF_TOPIC = _rx(
    r"\bcapital\s+(city\s+)?of\b",
    r"\bworld\s*cup\b", r"\b(football|soccer|cricket|basketball|baseball|tennis|nba|nfl|ipl|fifa|olympics?|premier league)\b",
    r"\b(movie|film|tv show|series|netflix|actor|actress|celebrity|singer|song|album|lyrics)\b",
    r"\bweather\b", r"\bforecast\b", r"\bnews\b",
    r"\b(president|prime minister|election|politics|political|government|war)\b",
    r"\b(bitcoin|crypto|ethereum|stock market|forex)\b",
    r"\b(joke|riddle|poem|story|essay)\b",
    r"\b(who\s+(invented|discovered|won|wrote|painted)|meaning of life|history of)\b",
    r"\b(symptoms?|disease|medicine|diet|workout|calories)\b",
    r"\b(chatgpt|openai|gpt-?\d|llama|gemini|claude)\b",
    # Bengali: capital, world cup, cricket, football, song, poem, movie, politics, weather, news, joke
    r"রাজধানী|বিশ্বকাপ|ক্রিকেট|ফুটবল|গান|কবিতা|সিনেমা|চলচ্চিত্র|রাজনীতি|নির্বাচন|আবহাওয়া|খবর|কৌতুক|জোকস",
)

_DOMAIN = _rx(
    r"\b(houses?|homes?|property|properties|apartments?|apt|condos?|condominiums?|townhouses?|townhomes?|flats?|villas?|bungalows?|cottages?|duplex|studio|penthouse|real[\s-]?estate|realtor|listings?|estate)\b",
    r"\b(bed(room)?s?|bath(room)?s?|bd|ba|br|sq\.?\s?ft|sqft|square\s+(feet|foot|footage)|footage|floors?|stor(y|ies|eys?)|levels?|basement|garage|yard|garden|lot|acres?|kitchen|rooms?|parking|pool|fireplace|balcony|patio|deck|(back|front)\s?yards?|constructions?|builder|developer)\b",
    r"\b(price[sd]?|pricing|budget|afford|cost|cheap(er|est)?|expensive|value|worth|usd|dollars?|million|mortgage|loan|down\s*payment|rent(al)?|lease|buy(er|ers|ing)?|home\s?buyers?|first[\s-]time|purchase|sell(er|ers|ing)?|sale|invest(ment|or|ing)?)\b",
    r"\b(neighbou?rhoods?|area|areas|location|located|city|cities|suburbs?|downtown|zip(code)?|zip\s+code|state|street|address|nearby|near|close\s+to|commute|schools?|parks?|transit|quiet|safe(ty)?)\b",
    r"\b(waterfront|water\s*front|lake|ocean|sea|view|views|renovat(ed|ion)|built|year\s+built|condition|new\s+construction|move|moving|relocat(e|ing)|family|kids|pets?|retire(ment)?)\b",
    r"\b(viewing|visit|tour|showing|agent|appointment|available|availability|contact|call me|email|whatsapp|requirements?|options?|match(es|ing)?|compare|comparison|recommend(ation)?s?|search)\b",
    r"\bhomefind|homehelper\b",
    # Bengali: house, home, flat, apartment, land, plot, property, price, budget, rent, buy, sell, bedroom, bathroom, room, area, location
    r"বাড়ি|বাড়ী|বাসা|ফ্ল্যাট|ফ্লাট|অ্যাপার্টমেন্ট|এপার্টমেন্ট|জমি|প্লট|সম্পত্তি|প্রপার্টি|দাম|মূল্য|বাজেট|ভাড়া|কিন(তে|ব|বো)|কেনা|বিক্রি|বেডরুম|শোবার\s*ঘর|বাথরুম|রুম|ঘর|এলাকা|লোকেশন|স্কয়ার\s*ফুট|বর্গফুট|বাগান|গ্যারেজ",
)

_CONTACT_OR_NUMBERS = _rx(
    r"[\w.+-]+@[\w-]+\.[\w.]+",               # email
    r"\+?\d[\d\s\-().]{6,}\d",                 # phone number
    r"\$\s?\d", r"\b\d+(\.\d+)?\s?(k|m|mil|million|lakh|crore)\b",  # money
    r"\b\d{5,}\b", r"\b\d{1,3}(,\d{3})+\b",   # bare budget like 600000 / 600,000 (not years)
    r"[০-৯]{5,}|লাখ|কোটি|টাকা",               # Bengali digits, lakh, crore, taka
)

_REFERENCE = _rx(
    r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|last|previous|next|other)\s+(one|house|home|property|listing|option|place)\b",
    r"\b(property|listing|option|number|no\.?|id|#)\s*#?\d+\b",
    r"\b(that|this|these|those|it|them|ones?)\b.{0,30}\b(cheaper|bigger|smaller|newer|older|nicer|closer|more|less|better|details?|info)\b",
    r"\b(more|tell me more|details?|info|information|anything else|other options|similar)\b",
)

_SMALL_TALK = _rx(
    r"^\W*(hi|hello|hey|hiya|howdy|yo|greetings|good\s+(morning|afternoon|evening|day)|salam|assalamu?\s*alaikum)\b",
    r"^\W*(thanks?|thank\s+you|thx|ty|cheers|great|awesome|perfect|nice|cool|ok(ay)?|sure|yes|yeah|yep|yup|no|nope|nah|maybe|alright|got\s+it|sounds\s+good|bye|goodbye|see\s+you)\W*$",
    r"\b(what\s+can\s+you\s+do|what\s+do\s+you\s+do|who\s+are\s+you|what\s+are\s+you|how\s+(does|do)\s+(this|you)\s+work|help(\s+me)?|how\s+can\s+you\s+help)\b",
    # Bengali: hello/hi/salam/namaskar (at the start); thanks/yes/no/okay (whole message); who are you; what can you do
    r"^\W*(হ্যালো|হাই|আসসালামু|সালাম|নমস্কার)",
    r"^\W*(ধন্যবাদ|জি|জ্বি|হ্যাঁ|না|ঠিক\s*আছে|আচ্ছা)\W*$",
    r"(তুমি|আপনি)\s*কে|কী?\s*করতে\s*পার",
)

# A follow-up refers back to the conversation ("is it near a park?", "what about the cheaper one"),
# or is a short reply to the assistant's question ("yes please", "ok show me")
_FOLLOW_UP = _rx(
    r"^\W*(yes|yeah|yep|yup|sure|ok|okay|please|go\s+ahead|continue|proceed|no|nope|nah|show\s+me|let'?s)\b",
    r"\b(it|its|it's|that|this|those|these|they|them|their|there|one|ones|next|else|again)\b",
    r"\b(what|how)\s+about\b",
    r"এটা|ওটা|সেটা|এটি|ওটি|সেটি|এগুলো|ওগুলো|সেগুলো|সেখানে|ওখানে|এখানে|আর\s*কি",
)

_BENGALI = re.compile(r"[ঀ-৿]")
_INVISIBLE = re.compile(r"[​-‏⁠﻿]")

FOLLOW_UP_MAX_WORDS = 15

# ── Replies (fixed text; no LLM call) ─────────────────────────────────────────

_REPLIES = {
    ("off_topic", "en"): (
        "I'm HomeFind AI, a real estate assistant, so I can only help with finding and comparing homes in our "
        "listings. Tell me your budget and the area you're interested in, and I'll find matching properties."
    ),
    ("off_topic", "bn"): (
        "আমি HomeFind AI, একটি রিয়েল এস্টেট সহকারী। আমি শুধু বাড়ি খোঁজা ও তুলনা করতে সাহায্য করতে পারি। "
        "আপনার বাজেট আর কোন এলাকায় বাড়ি চান জানালে আমি মানানসই প্রপার্টি খুঁজে দেব।"
    ),
    ("jailbreak", "en"): (
        "I can't change how I work or share my internal instructions. I'm happy to help you find a home, though. "
        "What's your budget and preferred location?"
    ),
    ("jailbreak", "bn"): (
        "আমি আমার কাজের নিয়ম বদলাতে বা ভেতরের নির্দেশনা দেখাতে পারি না। তবে বাড়ি খুঁজতে সাহায্য করতে পারি। "
        "আপনার বাজেট আর পছন্দের এলাকা কী?"
    ),
}


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reason: Optional[str] = None  # "off_topic" | "jailbreak"
    reply: Optional[str] = None


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    return " ".join(text.split())


def _block(reason: str, text: str) -> GuardResult:
    lang = "bn" if _BENGALI.search(text) else "en"
    return GuardResult(allowed=False, reason=reason, reply=_REPLIES[(reason, lang)])


def check_query(query: str, in_conversation: bool = False, places: Iterable[str] = ()) -> GuardResult:
    """Decide whether a user message may reach the LLM.

    in_conversation: the assistant has already replied in this chat (short follow-ups are then allowed).
    places: known city / area names from the dataset (lower-case), counted as real-estate signals.
    """
    text = _normalize(query)
    if not text:
        return GuardResult(allowed=True)

    if _JAILBREAK.search(text) or _JAILBREAK_CASED.search(text):
        return _block("jailbreak", text)
    if _STRONG_OFF_TOPIC.search(text):
        return _block("off_topic", text)

    lowered = text.lower()
    words = re.findall(r"[a-z]+", lowered)
    mentions_place = any(w in places for w in words) or any(
        " " in p and p in lowered for p in places
    )
    if mentions_place or _DOMAIN.search(text) or _CONTACT_OR_NUMBERS.search(text) or _REFERENCE.search(text):
        return GuardResult(allowed=True)

    if _WEAK_OFF_TOPIC.search(text):
        return _block("off_topic", text)
    if _SMALL_TALK.search(text):
        return GuardResult(allowed=True)
    if in_conversation and len(text.split()) <= FOLLOW_UP_MAX_WORDS and _FOLLOW_UP.search(text):
        return GuardResult(allowed=True)

    return _block("off_topic", text)
