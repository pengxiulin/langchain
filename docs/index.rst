欢迎来到 LangChain
==========================

LangChain is a framework for developing applications powered by language models. We believe that the most powerful and differentiated applications will not only call out to a language model via an API, but will also:

LangChain是一个用于开发由语言模型支持的应用程序框架。我们相信，最强大、最有差异的应用程序不仅会通过API调用语言模型，而且还会:

- *Be data-aware*: connect a language model to other sources of data
- *Be agentic*: allow a language model to interact with its environment

- *能够感知数据*: 将语言模型连接到其他数据源
- *有自觉性*: 允许语言模型与其环境交互

The LangChain framework is designed with the above principles in mind.

LangChain框架设计时考虑到了上述原则。

This is the Python specific portion of the documentation. For a purely conceptual guide to LangChain, see `here <https://docs.langchain.com/docs/>`_. For the JavaScript documentation, see `here <https://js.langchain.com/docs/>`_.

这是文档中特定于Python的部分。有关LangChain的纯概念指南，请参阅`这里 <https://docs.langchain.com/docs/>`_. JavaScript 文档, 请查看 `here <https://js.langchain.com/docs/>`_.

入门
----------------

Checkout the below guide for a walkthrough of how to get started using LangChain to create an Language Model application.

接下来我们开始来了解如何使用LangChain创建语言模型应用程序。

- `开始文档 <./getting_started/getting_started.html>`_

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :name: getting_started
   :hidden:

   getting_started/getting_started.md

Modules
-----------

LangChain提供支持的主要模块有若干个。
对于每个模块，我们都提供了一些入门示例、操作指南、参考文档和概念指南。
这些模块的复杂性依次为:

- `Models <./modules/models.html>`_: LangChain支持的各种模型类型和模型集成。

- `Prompts <./modules/prompts.html>`_: 包括prompt管理、优化和序列化.

- `Memory <./modules/memory.html>`_: 内存是在调用chain和agent之间保持状态。LangChain提供了一个内存的标准接口、一个内存实现的集合和chains/agents如何使用内存的示例。

- `Indexes <./modules/indexes.html>`_: 当与您自己的文本数据相结合时，语言模型通常更强大 —— Index模块涵盖了实现这一点的最佳实践。

- `Chains <./modules/chains.html>`_: Chains不仅仅是一个LLM调用，而是一系列调用（无论是对LLM还是对不同的实用程序）。LangChain为chains提供了标准接口，与其他工具进行了大量集成，并为常见应用程序提供了端到端链。



- `Agents <./modules/agents.html>`_: Agents涉及LLM决定采取哪些动作、执行该动作、查看观察结果，并重复该观察结果直到完成。LangChain给Agent提供标准接口、可供选择的代理选择以及端到端代理的示例。


.. toctree::
   :maxdepth: 1
   :caption: Modules
   :name: modules
   :hidden:

   ./modules/models.rst
   ./modules/prompts.rst
   ./modules/indexes.md
   ./modules/memory.md
   ./modules/chains.md
   ./modules/agents.md

用例
----------

上面说的模块可以通过多种方式使用。LangChain也在这方面提供指导和帮助。以下是LangChain支持的一些常见用例。

- `自主任务机器人(Autonomous Agents) <./use_cases/autonomous_agents.html>`_: 自主任务机器人是长时间运行的机器人，它们采取多个步骤来尝试实现目标。示例包括 AutoGPT 和 BabyAGI。


- `Agent Simulations <./use_cases/agent_simulations.html>`_: 将代理置于沙箱中并观察它们如何相互交互或与事件交互可能是观察其长期记忆能力的一种有趣方式。

- `个人助理 <./use_cases/personal_assistants.html>`_: LangChain的主要用例。个人助理需要执行任务、记住交互并了解您的数据。

- `问答 <./use_cases/question_answering.html>`_: LangChain第二大用例。回答针对特定文件的问题，仅利用这些文件中的信息来构建答案。

- `聊天机器人 <./use_cases/chatbots.html>`_: Since language models are good at producing text, that makes them ideal for creating chatbots.

- `查询表格数据 <./use_cases/tabular.html>`_: 如果您想了解如何使用LLM查询以表格格式（csv、SQL、大数据源等）存储的数据，可以看看这个页面。

- `理解代码(Code Understanding) <./use_cases/code.html>`_: 如果你想了解如何使用LLM从github查询源码，你应该阅读此页面。

- `调用API <./use_cases/apis.html>`_: 让LLM能够调用API将极大提升LLM获取最新信息并允许他们执行任务。

- `提取(Extraction) <./use_cases/extraction.html>`_: 从文本中提取结构化信息。

- `摘要总结 <./use_cases/summarization.html>`_: 把长文档总结成更短、更浓缩的信息块。一种数据增强生成。

- `评估 <./use_cases/evaluation.html>`_: 众所周知，生成模型很难用传统指标进行评估。评估它们的一种新方法是使用语言模型本身进行评估。LangChain提供了一些prompts/chains来协助这一点。


.. toctree::
   :maxdepth: 1
   :caption: Use Cases
   :name: use_cases
   :hidden:

   ./use_cases/personal_assistants.md
   ./use_cases/autonomous_agents.md
   ./use_cases/agent_simulations.md
   ./use_cases/question_answering.md
   ./use_cases/chatbots.md
   ./use_cases/tabular.rst
   ./use_cases/code.md
   ./use_cases/apis.md
   ./use_cases/summarization.md
   ./use_cases/extraction.md
   ./use_cases/evaluation.rst


参考文档
---------------

LangChain的所有参考文档，都在这里：关于LangChain的所有方法、类、安装方法和集成设置的完整文档。


- `Reference Documentation <./reference.html>`_
.. toctree::
   :maxdepth: 1
   :caption: Reference
   :name: reference
   :hidden:

   ./reference/installation.md
   ./reference/integrations.md
   ./reference.rst


LangChain生态
-------------------

其他公司/产品如何与 LangChain 一起使用的指南

- `LangChain 生态系统 <./ecosystem.html>`_

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Ecosystem
   :name: ecosystem
   :hidden:

   ./ecosystem.rst


其他资源
---------------------

一些在开发应用程序时可能有用的其他资源集合！

- `LangChainHub <https://github.com/hwchase17/langchain-hub>`_: LangChainHub 是一个分享和探索其他提示、链和代理的地方。

- `术语表 <./glossary.html>`_: 所有相关术语、论文、方法等的词汇表，无论是否在LangChain中实现！

- `优秀项目案例 <./gallery.html>`_: 我们最喜欢的使用LangChain的项目的集合。对于寻找灵感或查看其他应用程序中的工作方式很有用。

- `部署 <./deployments.html>`_: 部署LangChain的指令、代码片段和模板存储库的集合。

- `Tracing <./tracing.html>`_: 在LangChain中可视化跟踪chain和proxy执行过程。

- `模型试验室 <./model_laboratory.html>`_: 试验不同的提示、模型和链是开发最佳应用程序的重要部分。ModelLaboratory 可以轻松做到这一点。

- `Discord <https://discord.gg/6adMQxSpJS>`_: 加入我们的 Discord，讨论 LangChain 的所有事情！

- `YouTube <./youtube.html>`_: LangChain教程和视频的集合。

- `生产支持 <https://forms.gle/57d8AmXBYp8PP8tZA>`_: 当您将 LangChains 投入生产时，我们很乐意提供更全面的支持。请填写此表格，我们将建立专门的支持 Slack 渠道。


.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :name: resources
   :hidden:

   LangChainHub <https://github.com/hwchase17/langchain-hub>
   ./glossary.md
   ./gallery.rst
   ./deployments.md
   ./tracing.md
   ./use_cases/model_laboratory.ipynb
   Discord <https://discord.gg/6adMQxSpJS>
   ./youtube.md
   Production Support <https://forms.gle/57d8AmXBYp8PP8tZA>
