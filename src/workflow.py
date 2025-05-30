from langgraph.graph import StateGraph, START, END
from typing import List, Annotated, Literal
from dataclasses import dataclass, field
import operator

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

# Agent step stubs


def supervisor(state: WorkflowState) -> WorkflowState:
    print("Supervisor called.")
    state.log.append('Supervisor')
    # Decide which agent to call next
    if state.step == "from_judge":
        # If last judge result was not reasonable and assessor_calls < 3, go to Assessor
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
    state.rag_calls += 1
    if state.rag_calls >= 3:
        state.rag_done = True
    return state


def assessor(state: WorkflowState) -> WorkflowState:
    print("Assessor agent called.")
    state.log.append('Assessor')
    state.assessor_calls += 1
    if state.assessor_calls >= 3:
        state.assessor_done = True
    return state


def llmasajudge(state: WorkflowState) -> WorkflowState:
    print("LLMAsAJudge agent called.")
    state.log.append('LLMAsAJudge')
    state.judge_calls += 1
    # Simulate alternation: first call not reasonable, then reasonable, then not reasonable, etc.
    if state.judge_calls % 2 == 1:
        state.judge_last_result = "not_reasonable"
    else:
        state.judge_last_result = "reasonable"
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
    # After LLMAsAJudge, always go to Supervisor, but mark the step
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
    print(
        f"RAG calls: {result['rag_calls']}, Assessor calls: {result['assessor_calls']}, Judge calls: {result['judge_calls']}")
