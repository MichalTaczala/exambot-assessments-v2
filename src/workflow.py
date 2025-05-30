import operator
import random
import time
from typing import List, Annotated, Literal
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, START, END

try:
    from agents.rag_agent import RAGAgent
    from agents.assessor_agent import AssessorAgent
    from agents.judge_agent import LLMAsAJudgeAgent
    from src.agent_call_manager import AgentCallManager
except ModuleNotFoundError:
    from agents.rag_agent import RAGAgent
    from agents.assessor_agent import AssessorAgent
    from agents.judge_agent import LLMAsAJudgeAgent
    from agent_call_manager import AgentCallManager

# Define the minimal state schema


@dataclass
class WorkflowState:
    log: Annotated[List[str], operator.add] = field(default_factory=list)
    rag_calls: int = 0
    assessor_calls: int = 0
    judge_calls: int = 0
    rag_done: bool = False
    assessor_done: bool = False
    judge_done: bool = False
    step: str = "start"
    judge_last_result: str = ""
    failures: Annotated[List[str], operator.add] = field(default_factory=list)
    retries: int = 0
    max_retries: int = 2
    last_agent: str = ""
    call_logs: Annotated[List[str], operator.add] = field(default_factory=list)


# Use the robust AgentCallManager
manager = AgentCallManager(max_retries=2, backoff=0.5)
manager.rag = RAGAgent()
manager.assessor = AssessorAgent()
manager.judge = LLMAsAJudgeAgent()

# Agent step wrappers


def supervisor(state: WorkflowState) -> WorkflowState:
    print("Supervisor called.")
    state.log.append('Supervisor')
    # Decide which agent to call next
    if state.step == "from_judge":
        if state.judge_last_result == "not_reasonable" and state.assessor_calls < 3:
            state.step = "to_assessor"
        elif not state.judge_done:
            state.step = "to_judge"
        else:
            state.step = "end"
    elif not state.rag_done:
        state.step = "to_rag"
    elif not state.assessor_done:
        state.step = "to_assessor"
    elif not state.judge_done:
        state.step = "to_judge"
    else:
        state.step = "end"
    return state


def rag(state: WorkflowState) -> WorkflowState:
    print("RAG agent called.")
    state.log.append('RAG')
    state.last_agent = 'RAG'
    try:
        result = manager.call_rag("What is Knowledge Representation and Reasoning?")
        state.log.append(f'RAG result: {result}')
        state.call_logs.append(str(manager.logs[-1]))
    except Exception as e:
        state.failures.append(f'RAG failed: {e}')
        state.rag_done = True
        return state
    state.rag_calls += 1
    if state.rag_calls >= 3:
        state.rag_done = True
    return state


def assessor(state: WorkflowState) -> WorkflowState:
    print("Assessor agent called.")
    state.log.append('Assessor')
    state.last_agent = 'Assessor'
    try:
        result = manager.call_assessor(
            "What is Knowledge Representation and Reasoning?",
            "Knowledge Representation and Reasoning (KRR) is a field of AI focused on representing information about the world in a form that a computer system can utilize to solve complex tasks.",
            "It is about how computers can store and use knowledge to solve problems."
        )
        state.log.append(f'Assessor result: {result}')
        state.call_logs.append(str(manager.logs[-1]))
    except Exception as e:
        state.failures.append(f'Assessor failed: {e}')
        state.assessor_done = True
        return state
    state.assessor_calls += 1
    if state.assessor_calls >= 3:
        state.assessor_done = True
    return state


def llmasajudge(state: WorkflowState) -> WorkflowState:
    print("LLMAsAJudge agent called.")
    state.log.append('LLMAsAJudge')
    state.last_agent = 'LLMAsAJudge'
    try:
        result = manager.call_judge(
            "What is Knowledge Representation and Reasoning?",
            "Knowledge Representation and Reasoning (KRR) is a field of AI focused on representing information about the world in a form that a computer system can utilize to solve complex tasks.",
            "It is about how computers can store and use knowledge to solve problems.",
            8,
            "Good explanation of the basic concept, but could be improved by mentioning that KRR involves not just storing knowledge but also reasoning with it to solve complex tasks."
        )
        state.log.append(f'Judge result: {result}')
        state.call_logs.append(str(manager.logs[-1]))
        if isinstance(result, dict) and result.get("reasonable") is not None:
            if result["reasonable"]:
                state.judge_last_result = "reasonable"
            else:
                state.judge_last_result = "not_reasonable"
    except Exception as e:
        state.failures.append(f'Judge failed: {e}')
        state.judge_done = True
        return state
    state.judge_calls += 1
    if state.judge_calls >= 3:
        state.judge_done = True
    return state


def supervisor_conditional(state: WorkflowState) -> Literal['RAG', 'Assessor', 'LLMAsAJudge', END]:
    if state.step == "to_rag":
        return 'RAG'
    elif state.step == "to_assessor":
        return 'Assessor'
    elif state.step == "to_judge":
        return 'LLMAsAJudge'
    else:
        return END


def rag_conditional(state: WorkflowState) -> Literal['Supervisor', END]:
    return 'Supervisor'


def assessor_conditional(state: WorkflowState) -> Literal['Supervisor', END]:
    return 'Supervisor'


def judge_conditional(state: WorkflowState) -> Literal['Supervisor', END]:
    state.step = "from_judge"
    return 'Supervisor'

# Define the workflow structure


def build_workflow():
    g = StateGraph(state_schema=WorkflowState)
    g.add_node('Supervisor', supervisor)
    g.add_node('RAG', rag)
    g.add_node('Assessor', assessor)
    g.add_node('LLMAsAJudge', llmasajudge)
    # Entry edge
    g.add_edge(START, 'Supervisor')
    # Supervisor decides which agent to call next or to end
    g.add_conditional_edges('Supervisor', supervisor_conditional, ['RAG', 'Assessor', 'LLMAsAJudge', END])
    # Each agent always returns to Supervisor
    g.add_conditional_edges('RAG', rag_conditional, ['Supervisor'])
    g.add_conditional_edges('Assessor', assessor_conditional, ['Supervisor'])
    g.add_conditional_edges('LLMAsAJudge', judge_conditional, ['Supervisor'])
    # End edge
    g.add_edge('Supervisor', END)
    return g


if __name__ == "__main__":
    workflow = build_workflow()
    executable = workflow.compile()
    state = WorkflowState()
    result = executable.invoke(state)
    print("Workflow log:", result["log"])
    print(f"Failures: {result['failures']}")
    print(f"Call logs: {result['call_logs']}")
    print(
        f"RAG calls: {result['rag_calls']}, Assessor calls: {result['assessor_calls']}, Judge calls: {result['judge_calls']}")
    print(f"Retries on last call: {result['retries']}")
