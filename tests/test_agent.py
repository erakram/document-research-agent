import json

import pytest

from app.agent import AgentStepLimitError, DocumentResearchAgent
from tests.conftest import FakeGroq, llm_message, tool_call


def make_agent(store, responses, max_steps=6):
    client = FakeGroq(responses)
    return DocumentResearchAgent(client, store, "test-model", max_steps), client


def search_result(client, index=-1):
    tool_msg = [m for m in client.requests[index]["messages"] if m["role"] == "tool"][-1]
    return json.loads(tool_msg["content"])


def test_search_then_answer_with_citation(filled_store):
    meals_id = filled_store.search("meals travel")[0].chunk_id
    agent, client = make_agent(filled_store, [
        llm_message(tool_calls=[tool_call("c1", "search_documents", {"query": "meals travel"})]),
        llm_message(f"Meals are covered up to USD 60 per day [S{meals_id}]."),
    ])
    reply = agent.ask("What's the meal allowance?")

    passages = search_result(client)["passages"]
    assert passages[0]["source_id"] == f"S{meals_id}"
    assert [c.source_id for c in reply.citations] == [f"S{meals_id}"]
    assert reply.citations[0].document == "handbook.md"


def test_made_up_citations_are_dropped(filled_store):
    meals_id = filled_store.search("meals travel")[0].chunk_id
    agent, _ = make_agent(filled_store, [
        llm_message(tool_calls=[tool_call("c1", "search_documents", {"query": "meals travel"})]),
        llm_message(f"USD 60 [S{meals_id}], and something invented [S9999]."),
    ])
    assert [c.source_id for c in agent.ask("q").citations] == [f"S{meals_id}"]


def test_grouped_and_repeated_citations(filled_store):
    ids = [h.chunk_id for h in filled_store.search("employees budget remotely meals", top_k=3)]
    a, b = ids[0], ids[1]
    agent, _ = make_agent(filled_store, [
        llm_message(tool_calls=[tool_call("c1", "search_documents", {"query": "employees budget remotely meals"})]),
        llm_message(f"First [S{a}, S{b}]. Again [S{a}]."),
    ])
    assert [c.source_id for c in agent.ask("q").citations] == [f"S{a}", f"S{b}"]


def test_full_width_citations_are_normalized(filled_store):
    meals_id = filled_store.search("meals travel")[0].chunk_id
    agent, _ = make_agent(filled_store, [
        llm_message(tool_calls=[tool_call("c1", "search_documents", {"query": "meals travel"})]),
        llm_message(f"USD 60 per day 【S{meals_id}†L1-L2】."),
    ])
    reply = agent.ask("q")
    assert reply.answer == f"USD 60 per day [S{meals_id}]."
    assert [c.source_id for c in reply.citations] == [f"S{meals_id}"]


def test_no_results_tells_the_llm(filled_store):
    agent, client = make_agent(filled_store, [
        llm_message(tool_calls=[tool_call("c1", "search_documents", {"query": "zebra volcano"})]),
        llm_message("I couldn't find this in the documents."),
    ])
    reply = agent.ask("Tell me about zebras")
    result = search_result(client)
    assert result["passages"] == [] and "No relevant passages" in result["note"]
    assert reply.citations == []


def test_multiple_searches_accumulate_sources(filled_store):
    meals = filled_store.search("meals travel")[0].chunk_id
    plan = filled_store.search("business plan device month")[0].chunk_id
    agent, _ = make_agent(filled_store, [
        llm_message(tool_calls=[tool_call("c1", "search_documents", {"query": "meals travel"})]),
        llm_message(tool_calls=[tool_call("c2", "search_documents", {"query": "business plan device month"})]),
        llm_message(f"Meals: USD 60 [S{meals}]. Plan: USD 15 [S{plan}]."),
    ])
    reply = agent.ask("Two questions")
    assert len(reply.tool_calls) == 2 and len(reply.citations) == 2


def test_list_documents_tool(filled_store):
    agent, client = make_agent(filled_store, [
        llm_message(tool_calls=[tool_call("c1", "list_documents", {})]),
        llm_message("You have two documents."),
    ])
    agent.ask("What documents do I have?")
    assert {d["name"] for d in search_result(client)["documents"]} == {"handbook.md", "product.pdf"}


@pytest.mark.parametrize("raw, expected", [
    ("{bad json", "not valid JSON"),
    (json.dumps({"query": ""}), "non-empty"),
])
def test_bad_tool_arguments_return_errors(filled_store, raw, expected):
    agent, client = make_agent(filled_store, [
        llm_message(tool_calls=[tool_call("c1", "search_documents", raw)]),
        llm_message("Sorry."),
    ])
    agent.ask("q")
    assert expected in search_result(client)["error"]


def test_last_step_forces_an_answer(filled_store):
    agent, client = make_agent(filled_store, [
        llm_message(tool_calls=[tool_call("c1", "search_documents", {"query": "stock options"})]),
        llm_message("I couldn't find this in the documents."),
    ], max_steps=2)
    reply = agent.ask("Stock option policy?")
    assert reply.answer == "I couldn't find this in the documents."
    assert client.requests[0]["tool_choice"] == "auto"
    assert "tools" not in client.requests[1]  # last step: no tools offered
    assert client.requests[1]["messages"][-1]["role"] == "system"


def test_reasoning_effort_only_for_reasoning_models():
    from app.agent import model_options
    assert model_options("openai/gpt-oss-120b") == {"reasoning_effort": "low"}
    assert model_options("llama-3.3-70b-versatile") == {}


def test_repeated_passages_are_not_resent(filled_store):
    agent, client = make_agent(filled_store, [
        llm_message(tool_calls=[tool_call("c1", "search_documents", {"query": "meals travel"})]),
        llm_message(tool_calls=[tool_call("c2", "search_documents", {"query": "meals travel"})]),
        llm_message("Done."),
    ])
    agent.ask("q")
    second = search_result(client)["passages"]
    assert second and all(p == {"source_id": p["source_id"], "note": "already shown above"} for p in second)


def test_step_limit(filled_store):
    loop = [llm_message(tool_calls=[tool_call(f"c{i}", "list_documents", {})]) for i in range(2)]
    agent, _ = make_agent(filled_store, loop, max_steps=2)
    with pytest.raises(AgentStepLimitError):
        agent.ask("q")
