import os
import logging
import typing
from typing import Optional

from tools.contacts import get_all_contacts
from tools.vocode import call_phone_number
from tools.word_of_the_day import word_of_the_day
#from replit_config_manager import ReplitConfigManager
from call_transcript_utils import add_transcript

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

from vocode.streaming.models.events import Event, EventType
from vocode.streaming.models.transcript import TranscriptCompleteEvent
from vocode.streaming.utils import events_manager
from vocode.streaming.telephony.server.base import TelephonyServer

app = FastAPI(docs_url=None)
templates = Jinja2Templates(directory="templates")

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


class EventsManager(events_manager.EventsManager):

  def __init__(self):
    super().__init__(subscriptions=[EventType.TRANSCRIPT_COMPLETE])

  def handle_event(self, event: Event):
    if event.type == EventType.TRANSCRIPT_COMPLETE:
      transcript_complete_event = typing.cast(TranscriptCompleteEvent, event)
      add_transcript(
        transcript_complete_event.conversation_id,
        transcript_complete_event.transcript.to_string(),
      )


config_manager = ReplitConfigManager()

REPLIT_URL = f"{os.getenv('REPL_SLUG')}.{os.getenv('REPL_OWNER')}.repl.co"

telephony_server = TelephonyServer(base_url=REPLIT_URL,
                                   config_manager=config_manager,
                                   inbound_call_configs=[],
                                   events_manager=EventsManager(),
                                   logger=logger)


@app.get("/")
async def main(request: Request):
  env_vars = {
    "REPLIT_URL": REPLIT_URL,
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    "DEEPGRAM_API_KEY": os.environ.get("DEEPGRAM_API_KEY"),
    "TWILIO_ACCOUNT_SID": os.environ.get("TWILIO_ACCOUNT_SID"),
    "TWILIO_AUTH_TOKEN": os.environ.get("TWILIO_AUTH_TOKEN"),
    "ELEVEN_LABS_API_KEY": os.environ.get("ELEVEN_LABS_API_KEY"),
    "OUTBOUND_CALLER_NUMBER": os.environ.get("OUTBOUND_CALLER_NUMBER"),
  }

  return templates.TemplateResponse("index.html", {
    "request": request,
    "env_vars": env_vars
  })


@app.post("/start")
async def start_agent(objective: Optional[str] = Form(None)):
  try:
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)
    agent = initialize_agent(
      tools=[get_all_contacts, call_phone_number, word_of_the_day],
      llm=llm,
      agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
      verbose=True,
      memory=memory,
    )
    agent.run(objective)
  except Exception as e:
    logger.error(e)
    return {"error": str(e)}
  return {"status": "success"}


app.include_router(telephony_server.get_router())
