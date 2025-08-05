from typing import TypedDict, List, Annotated

class BaseMessages(TypedDict):
    messages : List = []
    history : List = []
    video_path : str = None
    response : List = []
    video_context : List = []
    need_summarizer : int 
    events : List= []
    need_events : int