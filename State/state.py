from typing import TypedDict, List, Annotated, Optional
    
class BaseMessages(TypedDict):
    messages: Annotated[List[str], "List of message strings"]
    history: Annotated[List[str], "List of history strings"]
    video_path: Annotated[Optional[str], "Path to the video, or None"]
    response: Annotated[List[str], "List of response strings"]
    video_context: Annotated[List[str], "Contextual information from video"]
    need_summarizer: Annotated[int, "Flag to indicate summarizer needed"]
    events: Annotated[List[str], "List of events"]
    need_events: Annotated[int, "Flag to indicate events needed"]