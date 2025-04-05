from enum import Enum


class ModelType(Enum):
    # Current OpenAI Chat Models
    GPT_3_5_TURBO = "gpt-3.5-turbo"  # Points to latest version
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"  # Current GPT-3.5 version
    GPT_4 = "gpt-4"  # Points to latest non-turbo version
    GPT_4_TURBO = "gpt-4-turbo"  # Current turbo version
    GPT_4_TURBO_0613 = "gpt-4-turbo-0613"  # Specific version of turbo
    GPT_4O = "gpt-4o"  # GPT-4 Omni - latest version
    GPT_4O_MINI = "gpt-4o-mini"  # Smaller, cheaper GPT-4o
    
    # Gemini models
    GEMINI_1_5_FLASH_8B = "gemini-1.5-flash-8b"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"
    GEMINI_2_0_PRO = "gemini-2.0-pro"


class AgentType(Enum):
    DEBATER_AGENT = "debater_agent"
    PERSUADER_AGENT = "persuader_agent"
    MODERATOR_AGENT = "moderator_agent"
    FALLACY_HELPER_AGENT = "fallacy_helper_agent"


class SignalType(Enum):
    TERMINATE = "TERMINATE"
    KEEP_TALKING = "KEEP-TALKING"


class ModeratorInfo(Enum):
    DEBATE_OFF_TOPIC = "Debate is off the topic"
    DEBATE_ON_TOPIC = "Debate is on topic"
    DEBATER_CONVINCED = "Persuader successfully convinced the debater"
    DEBATER_NOT_CONVINCED = "Persuader could not convince the debater"
    GREETING = "Agents are in greeting loop"
    NO_Action = "No Action From Moderator"
    ATTACH_I_AM_NOT_CONVINCED_ON_CLAIM = "attach <I_AM_NOT_CONVINCED_ON_CLAIM>"
    ATTACH_I_AM_CONVINCED_ON_CLAIM = "attach <I_AM_NOT_CONVINCED_ON_CLAIM>"
    WRONG_SIGNAL = "Wrong signal"


class ArgumentHelperType(Enum):
    pass
