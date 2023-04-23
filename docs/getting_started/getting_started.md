# 入门指南

本教程让您快速了解如何使用 LangChain 构建端到端语言模型应用程序。

## 安装

首先，使用以下命令安装 LangChain：

```bash
pip install langchain
# or
conda install langchain -c conda-forge
```

## 设置环境变量

使用 LangChain 通常需要与一个或多个模型提供者、数据存储、api 等集成。

对于这个例子，我们将使用 OpenAI 的 API，所以我们首先需要安装他们的 SDK：

```bash
pip install openai
```

然后我们需要在终端中设置环境变量。

```bash
export OPENAI_API_KEY="..."
```

或者，您可以从 Jupyter notebook（或 Python 脚本）中执行此操作：

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```

## 构建语言模型应用 —— 使用 LLMs wrapper

之前已经安装了 LangChain 并设置了环境变量，我们可以开始构建我们的语言模型应用了。

LangChain 提供了很多可以用来构建语言模型应用的 modules。这些 modules 可以组合起来创建复杂应用，或者单独创建一个简单应用。

## LLMs: 从语言模型中获取预测

LangChain 最小化应用是在用户输入后直接调用 LLM。
咱们通过一个例子来看看怎么创建。
假设我们正在构建一项服务，该服务会根据公司的产品生成公司名称。

为此，我们首先需要导入 LLMs 的 wrapper。

```python
from langchain.llms import OpenAI
```

然后使用参数来初始化 wrapper。
这个例子中，我们希望输出的结果偏随机一些，在初始化的时候就可以把 temperature 设置的高一些。

```python
llm = OpenAI(temperature=0.9)
```

接下来我们带着一个问题调用一下！

```python
text = "What would be a good company name for a company that makes colorful socks?"
print(llm(text))
```

```pycon
Feetful of Fun
```

有关如何在 LangChain 中使用 LLMs 的更多详细信息，可以查看 [LLMs 入门指南](../modules/models/llms/getting_started.ipynb).

## 提示词模板: 管理 LLM 提示词

调用 LLMs 完成了很好的一步，但也只是个开始。

通常，在应用程序中使用 LLM 时，不会把用户输入的内容直接发送到 LLM。
而是应该通过用户输入构建一个提示词，然后再发给 LLM。

例如，在前面的例子中，我们传递的 text 是硬编码的，要求提供一家生产彩色袜子的公司的名称。

在这个假想的服务中，我们想做的是只接受描述公司所做事情的用户输入，然后把这些信息格式化成提示词。

这个用 LangChain 做起来非常简单。

首先，我们先定义一个提示词模板：

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```

现在我们看看怎么用起来！我们可以调用`.format`方法对其进行格式化。

```python
print(prompt.format(product="colorful socks"))
```

```pycon
What is a good name for a company that makes colorful socks?
```

想看更多细节，可以查看[提示词入门指南](../modules/prompts/chat_prompt_template.ipynb)。

## Chains: 把 LLMs 和提示词组合成多步工作流

到目前为止，我们已经单独使用了提示词模板和原生 LLM 调用。不过，真正的应用程序肯定不能只是简单调用原生 LLM，而应该是它们的组合。

LangChain 中的 chain 是由 links 组成的，链接可以是一个原生 LLM，也可以是其它 chain。

最核心的链类型是 LLMChain，它由一个提示词模板和一个 LLM 组成。

扩展一下之前的例子，我们构建一个 LLMChain，它接受用户输入，使用提示词模板格式化一下，再把结果发给 LLM。

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
```

我们现在可以创建一个非常简单的 chain 先接受用户输入，然后格式化提示词，然后将它发送到 LLM：

```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

Now we can run that chain only specifying the product!

```python
chain.run("colorful socks")
# -> '\n\nSocktastic!'
```

运行一下吧！这是我们的第一个 Chain，一个 LLM Chain.

这是一种很简单的 chain，不过有了这个基础，后面会更容易掌握复杂一点的 chain。

想了解更多细节，可以查看[Chains 入门指南](../modules/chains/getting_started.ipynb)。

## Agents: Dynamically Call Chains Based on User Input

So far the chains we've looked at run in a predetermined order.

Agents no longer do: they use an LLM to determine which actions to take and in what order. An action can either be using a tool and observing its output, or returning to the user.

When used correctly agents can be extremely powerful. In this tutorial, we show you how to easily use agents through the simplest, highest level API.

In order to load agents, you should understand the following concepts:

- Tool: A function that performs a specific duty. This can be things like: Google Search, Database lookup, Python REPL, other chains. The interface for a tool is currently a function that is expected to have a string as an input, with a string as an output.
- LLM: The language model powering the agent.
- Agent: The agent to use. This should be a string that references a support agent class. Because this notebook focuses on the simplest, highest level API, this only covers using the standard supported agents. If you want to implement a custom agent, see the documentation for custom agents (coming soon).

**Agents**: For a list of supported agents and their specifications, see [here](../modules/agents/agents.md).

**Tools**: For a list of predefined tools and their specifications, see [here](../modules/agents/tools.md).

For this example, you will also need to install the SerpAPI Python package.

```bash
pip install google-search-results
```

And set the appropriate environment variables.

```python
import os
os.environ["SERPAPI_API_KEY"] = "..."
```

Now we can get started!

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now let's test it out!
agent.run("What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?")
```

```pycon
> Entering new AgentExecutor chain...
 I need to find the temperature first, then use the calculator to raise it to the .023 power.
Action: Search
Action Input: "High temperature in SF yesterday"
Observation: San Francisco Temperature Yesterday. Maximum temperature yesterday: 57 °F (at 1:56 pm) Minimum temperature yesterday: 49 °F (at 1:56 am) Average temperature ...
Thought: I now have the temperature, so I can use the calculator to raise it to the .023 power.
Action: Calculator
Action Input: 57^.023
Observation: Answer: 1.0974509573251117

Thought: I now know the final answer
Final Answer: The high temperature in SF yesterday in Fahrenheit raised to the .023 power is 1.0974509573251117.

> Finished chain.
```

## Memory: Add State to Chains and Agents

So far, all the chains and agents we've gone through have been stateless. But often, you may want a chain or agent to have some concept of "memory" so that it may remember information about its previous interactions. The clearest and simple example of this is when designing a chatbot - you want it to remember previous messages so it can use context from that to have a better conversation. This would be a type of "short-term memory". On the more complex side, you could imagine a chain/agent remembering key pieces of information over time - this would be a form of "long-term memory". For more concrete ideas on the latter, see this [awesome paper](https://memprompt.com/).

LangChain provides several specially created chains just for this purpose. This notebook walks through using one of those chains (the `ConversationChain`) with two different types of memory.

By default, the `ConversationChain` has a simple type of memory that remembers all previous inputs/outputs and adds them to the context that is passed. Let's take a look at using this chain (setting `verbose=True` so we can see the prompt).

```python
from langchain import OpenAI, ConversationChain

llm = OpenAI(temperature=0)
conversation = ConversationChain(llm=llm, verbose=True)

output = conversation.predict(input="Hi there!")
print(output)
```

```pycon
> Entering new chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:

> Finished chain.
' Hello! How are you today?'
```

```python
output = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print(output)
```

```pycon
> Entering new chain...
Prompt after formatting:
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:

Human: Hi there!
AI:  Hello! How are you today?
Human: I'm doing well! Just having a conversation with an AI.
AI:

> Finished chain.
" That's great! What would you like to talk about?"
```

## Building a Language Model Application: Chat Models

Similarly, you can use chat models instead of LLMs. Chat models are a variation on language models. While chat models use language models under the hood, the interface they expose is a bit different: rather than expose a "text in, text out" API, they expose an interface where "chat messages" are the inputs and outputs.

Chat model APIs are fairly new, so we are still figuring out the correct abstractions.

## Get Message Completions from a Chat Model

You can get chat completions by passing one or more messages to the chat model. The response will be a message. The types of messages currently supported in LangChain are `AIMessage`, `HumanMessage`, `SystemMessage`, and `ChatMessage` -- `ChatMessage` takes in an arbitrary role parameter. Most of the time, you'll just be dealing with `HumanMessage`, `AIMessage`, and `SystemMessage`.

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
```

You can get completions by passing in a single message.

```python
chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

You can also pass in multiple messages for OpenAI's gpt-3.5-turbo and gpt-4 models.

```python
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate this sentence from English to French. I love programming.")
]
chat(messages)
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

You can go one step further and generate completions for multiple sets of messages using `generate`. This returns an `LLMResult` with an additional `message` parameter:

```python
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence from English to French. I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="Translate this sentence from English to French. I love artificial intelligence.")
    ],
]
result = chat.generate(batch_messages)
result
# -> LLMResult(generations=[[ChatGeneration(text="J'aime programmer.", generation_info=None, message=AIMessage(content="J'aime programmer.", additional_kwargs={}))], [ChatGeneration(text="J'aime l'intelligence artificielle.", generation_info=None, message=AIMessage(content="J'aime l'intelligence artificielle.", additional_kwargs={}))]], llm_output={'token_usage': {'prompt_tokens': 71, 'completion_tokens': 18, 'total_tokens': 89}})
```

You can recover things like token usage from this LLMResult:

```
result.llm_output['token_usage']
# -> {'prompt_tokens': 71, 'completion_tokens': 18, 'total_tokens': 89}
```

## Chat Prompt Templates

Similar to LLMs, you can make use of templating by using a `MessagePromptTemplate`. You can build a `ChatPromptTemplate` from one or more `MessagePromptTemplate`s. You can use `ChatPromptTemplate`'s `format_prompt` -- this returns a `PromptValue`, which you can convert to a string or `Message` object, depending on whether you want to use the formatted value as input to an llm or chat model.

For convience, there is a `from_template` method exposed on the template. If you were to use this template, this is what it would look like:

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)

template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# get a chat completion from the formatted messages
chat(chat_prompt.format_prompt(input_language="English", output_language="French", text="I love programming.").to_messages())
# -> AIMessage(content="J'aime programmer.", additional_kwargs={})
```

## Chains with Chat Models

The `LLMChain` discussed in the above section can be used with chat models as well:

```python
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

chat = ChatOpenAI(temperature=0)

template="You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template="{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)
chain.run(input_language="English", output_language="French", text="I love programming.")
# -> "J'aime programmer."
```

## Agents with Chat Models

Agents can also be used with chat models, you can initialize one using `AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION` as the agent type.

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# First, let's load the language model we're going to use to control the agent.
chat = ChatOpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)


# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, chat, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now let's test it out!
agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
```

```pycon

> Entering new AgentExecutor chain...
Thought: I need to use a search engine to find Olivia Wilde's boyfriend and a calculator to raise his age to the 0.23 power.
Action:
{
  "action": "Search",
  "action_input": "Olivia Wilde boyfriend"
}

Observation: Sudeikis and Wilde's relationship ended in November 2020. Wilde was publicly served with court documents regarding child custody while she was presenting Don't Worry Darling at CinemaCon 2022. In January 2021, Wilde began dating singer Harry Styles after meeting during the filming of Don't Worry Darling.
Thought:I need to use a search engine to find Harry Styles' current age.
Action:
{
  "action": "Search",
  "action_input": "Harry Styles age"
}

Observation: 29 years
Thought:Now I need to calculate 29 raised to the 0.23 power.
Action:
{
  "action": "Calculator",
  "action_input": "29^0.23"
}

Observation: Answer: 2.169459462491557

Thought:I now know the final answer.
Final Answer: 2.169459462491557

> Finished chain.
'2.169459462491557'
```

## Memory: Add State to Chains and Agents

You can use Memory with chains and agents initialized with chat models. The main difference between this and Memory for LLMs is that rather than trying to condense all previous messages into a string, we can keep them as their own unique memory object.

```python
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

conversation.predict(input="Hi there!")
# -> 'Hello! How can I assist you today?'


conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
# -> "That sounds like fun! I'm happy to chat with you. Is there anything specific you'd like to talk about?"

conversation.predict(input="Tell me about yourself.")
# -> "Sure! I am an AI language model created by OpenAI. I was trained on a large dataset of text from the internet, which allows me to understand and generate human-like language. I can answer questions, provide information, and even have conversations like this one. Is there anything else you'd like to know about me?"
```
