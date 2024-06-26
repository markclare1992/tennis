# Plan
Below outlines a high level plan for work on the project.

## Exploratory Data Analysis

### Data Cleaning

#### Packages
- Use Jupyter Notebooks to explore the data and clean it up.
- Use Pandas for data manipulation.
- Use Pydantic to create a data validation framework.

#### Steps
- Explore the data CSV files and clean them up.
  - Identify any missing data and decide how to handle it.
  - Handle duplicate data.
  - Discard erroneous data.
  - Check IDs are unique.
  - Check for string similarity, i.e. can we match the same player/tournament with different names?
  - Outlier analysis. ?
- Note any additional data that may be needed, or "ideal" requirements of the data.
  - Identify possible external data sources that may be useful. MOVE TO ANOTHER SECTION
- Add a data validation framework to ensure data integrity.
  - This will be a set of rules that the data must pass before it can be used in the model.
  - This may be difficult due to lack of information, i.e. rules on the matches.

#### Extensions
- Create a formal data engineering pipeline to clean the data.
  - Use orchestration tools like Apache Airflow to schedule the pipeline.
- Save in a database (something like PSQL) or data warehouse (something like BigQuery).
- Schedule the pipeline to run at regular intervals.
- Process only new data, or data that has changed since the last run.
- Add suitable logging to the pipeline to track the data processing.

### Data Exploration

#### Packages
- Use Jupyter Notebooks to explore the data and clean it up.

#### Steps
- Explore the data to understand the relationships between the features.
  - Use correlation matrices to identify any relationships between the features.
- Identify any potential issues with the data.
  - Use outlier detection to identify any outliers in the data.
- Identify any possibly useful features that may be derived from the data.
  - Use feature engineering to create new features from the existing data.

### Data Visualization

#### Packages
- Use Jupyter Notebooks to explore the data.
  - Use histograms, scatter plots, box plots etc. to explore the data.
- Use Matplotlib, Seaborn, Plotly etc. to create visualizations.

#### Steps
- Create visualizations to help understand the data.
- Identify any relationships between the features.
  - Use heatmaps and pair plots to visualize correlations.

## Historical Feature Store
We need to be careful about data leakage, we only want to use data that would have been available at the time of the match.

### Packages
- Initially use pandas for data manipulation.
- Can use more advanced tools such as Spark for large datasets. (Should be able to use pandas methods in Spark)
- Ensure we have data for each match date.

## Model

### Packages
- Use Statsmodels for statistical models.
- Use Scikit-learn for standard machine learning models.
- Use Tensorflow, Pytorch etc. for deep learning models.
- Use Pymc, Pyro etc. for probabilistic programming.

### Modelling
- We have been asked to predict match outcome.
  - Treat as a prediction problem and assign probabilities to each outcome. (More useful than classification)
- Use basic regression models to predict the outcome.
- Add analytical Markov Chain Model
- Add Simulation Model that can expand upon the Markov Chain Model.
- Use observed data to obtain parameters for the model.
- Use more complex models (mixed effects) to obtain parameters.
- Use Bayesian models (PyMC, Pyro) to obtain parameters.
- Use traditional machine learning models with relevant dataset (See Historical Feature Store above.)
- Combine models to create an ensemble model.


### Evaluation

- Evaluate vs observed outcomes, what are we trying to minimize?
  - Define evaluation metrics based on the business objectives.
- Will depend on what markets you want to use the model to bet on. 
  - Use backtesting techniques for evaluating model performance on historical data. (Obtain this)
- Ideally would have market data from bookmakers you will likely use to bet on.
  - Explore possible sources of market data. (Oddsportal, Betfair, Oddschecker, etc.)
- Consider practical evaluation metrics, i.e.:
  - Speed of fitting/training/updating the model.
  - Speed of prediction.
  - Required live data for prediction.
- Can we describe performance of models, scenarios where they perform well, scenarios where they perform poorly, in an interpretable way?


## Productionisation

### Packages
- Can Use Docker to containerize the model.
- Can Use Kubernetes to orchestrate the containers.
- Can Use Flask, FastAPI etc. to create a REST API.
- Can use MlFlow for model tracking & model registry.
- Use Prometheus, Grafana, ELK stack etc. for monitoring.

### Deployment
- Deploy the model to a cloud service.
- Create a REST API to serve the model.
- Add monitoring to the model to track its performance.

### Monitoring
- Monitor the model to ensure it is performing as expected.
- Add alerting to the model to notify you if it is not performing as expected.
- Add logging to track the model's performance.
- Add a feedback loop to retrain the model if it is not performing as expected.

### Maintenance
- Regularly update the all steps.
- Regularly monitor underlying Sport, Rules, Players, Teams, Tournaments, etc.

### Security
- Ensure all parts of the project are secure.

### Scalability
- Ensure the model is scalable. (i.e., what if 20 tennis matches are played at the same time?)
  - Use horizontal scaling to scale the model and load balance the requests.
- Ensure the data is scalable.
  - Use a distributed database to store the data.

### Cost
- Ensure the model is cost-effective.
  - Use cost monitoring to track the cost of the model.
  - Use Spot resources, or schedule data pipelines to run at off-peak times.
  - Prediction Models and features need to be readily available.
- Ensure the data is cost-effective.
  - Where is data being procured from?
  - Where/how is data being stored?
  - Is there costs for retrieving & processing data?

### Additional Considerations
- Best Practices
  - Ensure the project follows best practices.
  - Ensure the project is well-documented.
  - Ensure the project is well-tested. 
    - Use Pytest to write tests for the project.
  - Ensure code is clean, has good variable names, docstrings, etc.
  - Use version control to track changes to the project, eg. Git.
  - Use packages like Ruff to ensure the quality of the code.
  - Use CI/CD to automate the testing and deployment of the project.
  - Use code reviews to ensure the quality of the code.
  - Use Confluence, Notion, etc. to document the project.
    - Possibly write formal documentation for the project, i.e, latex, Overleaf, etc.

