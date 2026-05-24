---
title: Initial Design
---

# Scope

- A data management tool for managing machine learning experiments
- A light weight experiment management solution
- Aimed at command-line users, which includes agents
- Simple folder-based structure enables synchronization via rsync
- Good support for interaction with an hpc cluster with very simple
  infrastructure, i.e. a main ssh node for access that is shared across
  all users and a pool of machines that get the files, run the job, and
  are torn down again
- wandb and tensorboard are easy to log to but difficult to export from
- They also add all sorts of features, like reports, that aren't
  necessary if you can use python
- Small to medium size and fast experimentation
- Compatibility with lightning
- Main selling points to me:
  - Analyze results in sql -\> Join results across different
    experiments, store only what's new, join the rest
  - Single file with clean data -\> Easy to store and share
  - Work with experiment objects instead of files
  - Deduplicating blob store -\> No need to worry about storing things
    twice and no need to worry about where to store it
  - Load artifacts by name
  - Load parts of your data in any format, e.g. csv
  - Load/query models by hyperparameters instead of file paths
- Future selling points:
  - Clear schemas enable agents to analyze your data
  - Group runs into collections by tags

# Details on the use case I have in mind

- One scripts per experiment (can be argparse subcommands)

# Changing database schemas

- Maybe maintain the schema versioned with the code so that you can run
  an earlier version
- They will change, e.g., because you now store one more thing in the
  experiment
- The problem is that I'm explicitly trying to keep it consistent in one
  database, so that has to be handled. Usually, you'd just create a new
  csv file
- In that case actually, initial table should be dropped and replaced,
  so no need for changing the schema and keeping it consistent with the
  old version
- Cases where that is truly needed may be rare, so maybe it's fine if
  they are expensive

# Missing features

- Avoid duplicated sql during analysis
- Plotting requires pandas and seaborn, maybe use grammar of plots for
  sql if it fits in nicely
- Continuous live logging, i.e. like tensorboard
- Ability to add load/store plugins, i.e. provide your own function to
  convert to bytes (otherwise have to stay consistent across PIL, numpy,
  torch, and growing) -\> Provide prebuild one that only requires a
  function to store bytes and also allow implementing a complete wrapper
  from scratch
- Avoid manually specifying the the schemas (maybe llms?) or make it an
  optional feature that gains power if used
- Need a method like `insert_pytorch_dataset`
- Generally need to make it easier to insert datasets
- Evolving database schemas
  - One table per subcommand keeps it more static, but it's still
    necessary

# Nice-to-have features

- Somehow maintain documentation automatically so that the llm can
  analyze the data for you
- Wrapper methods to avoid writing any sql if that is undesired, i.e.
  load the tables by name and join them automatically in the background

# Detailed features

- Artifacts, such as model checkpoints or images, are stored to a
  deduplicating blob store
- The database contains everything that isn't a blob, e.g.
  - Experiment metadata, such as hyperparameters
  - Experiment results, such as measured metrics
  - Tables for managing the blob store
  - Dataset labels
- Experiments and models are identified by unique ids
- Custom lightning logger (i.e. implement the interface from lightning)
- Every run gets an automatically generated timestamp when it is started
- The library provides a method `init_experiment` to register the
  experiment in the data (e.g. create a timestamp, etc.)
- After registering, a variety of methods allows storing data belonging
  to the experiment. There are two fundamental ones:
  - `store_table`
  - `store_artifact`

  Wrappers around these primitives can store multiple results together,
  e.g. a method `store_model` can take a model and call `store_table` to
  store hyperparameters and metrics and `store_artifact` to store a
  state<sub>dict</sub> as a model checkpoint.
- Experiments create parquet- or csv files that can be ingested into the
  main database
- Database constraints and pandera catch invalid data early
- Set the location through environment variables for portability and to
  use the right drive for each kind of data
- Load data as pytorch datasets without separate implementation
- Print statistics about the storage, e.g. how much space is taken and
  how much could be freed
- Dry runs
- Clean/compact blob store method

# Database model

- One database per project
- One table per experiment
- Runs and models get a unique id
- Every experiment (script or subcommand) gets a table
- General model table with all ids
- General artifacts table with all blob ids
- Separate out management tables e.g. those for blob storage, otherwise
  a bit ugly interface

# Designs to work out

## NEXT Database model

# Optimizations

## NEXT Avoid expensive imports by using bytes

# Research

## NEXT What makes a good spec?

## NEXT How to split up the work?
