# Work Done

## EDA - Initial Exploration
Used Jupyter Notebooks to explore the data in [initial_exploration](../notebooks/initial_exploration.ipynb).
Below is a summary of the main findings, please refer to the notebook for more details.

**Main Findings:**
- Match Info:
  - Contains court surface that will be useful.
  - Match length (3 or 5 sets) will impact win probability.
  - Messy Davis Cup data (exclude)
  - Possibly a lot of repeated tournament names (string similarity searches) that can be mapped.

- Match Outcome Stats:
  - Unstructured data, need to parse.
  - Messy with no clear structure.
  - Hacky parsing methods implemented that can be improved.
    - Still one row that can't be parsed (includes date)
  - Need to handle Retirements, Walkovers, and Disqualifications.
    - May want to exclude above from evaluation.
    - But some stats may still be useful (e.g. number of games won).
  - Minutes played might be useful for modelling fatigue or difference in quality.
    - Probably a proxy for number of points in each service game, and we do not have that data.

- Player Info:
  - Told player names are unique, so do not need to worry about string similarity.
  - Data looks sensible.
  - Some missing data for seed & rank, presumably for unseeded players.

- Player Stats:
  - Transform into a wide format.
  - Doesn't appear to be any missing data for any player stats.
  - After minor sanity check of some large data, seems fine.

## Data Processing Code

### Basic Processing
Extracted methods from the notebook into the `tennis/data_processing/basic_processing` package.
Files & methods should hopefully be self-documenting.

### Pydantic Validation
Started to write Pydantic models for the data, but not yet complete. (Limited value at this stage, and priorised other tasks)

### Historic Features
In order to prevent data leakage, we need to ensure features are calculated using only data available at the time of the match.
I.e. we filter for matches that occurred before the match we are calculating features for.

- Added hacky method to order and group and calculate some basic features.

### Pipelines
Added some basic code blocks to process the data, and merge together.
Data is saved as a CSV.


## EDA - Plots
Now we have a more useful data set, did a bit of EDA in [eda_plots](../notebooks/eda_plots.ipynb).
Could have spent more time on this, but wanted to get to the modelling stage.

Rolling Mean Plots are noisy, possibly should filter data until N> 10 games for each player?

## Modelling
### Models
Added Models Package.
- Added basic implementation of Markov Chain Model in markov_model.py. (Hopefully code is self-documenting)
  - Makes some assumptions to speed up coding process.
  - Easily extendable to more complex models.
  - Code quality & OOP easy to implement, but takes longer to write.

**Advantages**:
- Can be extended to in-play
- Can model other markets

**Disadvantages**:
- Assumes independence of points, which is not true.
- Assumes constant serve winning probabilities, which is not true.
- Difficult to extend to more complex models.

- Intended to add Simulator but didn't get to it.

Simulator easier to write & extend. Can use numba for speed if required.


### Model Training / Fitting Etc
Ideally would have liked to have a pipeline for this, but didn't get to it.


#### Logistic Regression
Simple model to compare against.

#### Basic Markov Model
Use player 1 avg prob of winning on serve % and player 2 avg prob of winning on serve %.
Plug into Markov Model to get match winning probs.

Issues: Performance poor, didn't have time to debug.

#### Player and Opponent Markov Model
Expand upon above and include player prob of winning on serve % and opponent allowed prob of winning on serve %.
I.e. Some information on both player serve ability and opponent return ability.
Take mean of both (i.e. player prob winning + opponent prob losing) / 2.

Issues: Similar to above, so didn't get to it.

#### Mixed Effects Markov Model
More formal model extending upon above.
Train mixed effects model to model serve winning probabilities from player and opponent.
Use these probabilities in Markov Model to get match winning probabilities.

Maximum likelihood operation failed to converge, didn't get time to investigate further.

#### Bayesian Model
Didn't get to this, but would be interesting to see how it compares to the above models.

(Use Hierarchical Model to model serve winning probabilities from player and opponent.)
Can extend in multiple ways, i.e., include surface, tournament level, age, player height etc.

#### Traditional ML Models
- Decision Tree
- Random Forest

### Evaluation
Used MlFlow to track experiments.
If we are comparing models they should be compared like for like, i.e., same training data and evaluation on fixtures they all have predictions for.
In practice any production model is likely to be trained on most recent data, and therefore should have a rolling evaluation.

Logged different performance metrics for each model.
- Log loss
- Brier Score

## Improvements/Extensions/Additional Data Etc
### Data
- Perform string similarity searches on entities, map to a standard name.
- Consider how we deal with incomplete matches.
- Implement Pydantic models for data, with strict validation rules linked to rules of tennis etc.
- Implement data pipelines (scheduling) and proper data processing and storage, i.e. DB or Data Warehouse.
- Obtain market odds data, ideally from bookmakers you will use to bet on.
- Obtain more detailed event data, i.e. point by point data, tracking data etc.
- Obtain tournament data, i.e. prize money, round money etc, indoor/outdoor etc.
- Revisit Feature Engineering
  - Match number proxy for fatigue?
  - Match minutes played proxy for fatigue?
  - How long since last match? How many games played in last match?
  - Travel, time zone changes etc.
  - Weather conditions, wind, humidity, light etc.
  - Crowd size and impact.
  - Is player at home country
  - Player Age/Height/Weight etc, **height useful for serve ability**
- Add logging etc.

### EDA
- Spend more time graphically exploring the data.

### Historic Features
- Improve the code to calculate historic features.
  - **Add time weighted features in addition to standard Mean/Median features**
  - Re-write in resilient way.
  - Use spark instead of pandas.
  

### Modelling
- Add model to predict retirements, walkovers, disqualifications etc.
  - I.e, possibly a player injury model.
- Improve Markov Model Code.
- Implement Simulator Model.
- Implement Player and Opponent Markov Model.
- Implement Mixed Effects Markov Model.
- Implement Bayesian Model.
  - Allow players skill to change over time.
  - Allow surface specific skill.
  - Allow stylistic interactions between players, i.e. Serve and Volley vs Baseline.
- Implement Traditional ML Models.
- Implement Ensemble Model.
- Can we add above onto market odds to get a more accurate prediction?

### Evaluation
- Create formal evaluation framework. (Possibly mlflow using DB + S3)
- Link to historical feature store.
- Compare models like for like.
- Evaluate models on a rolling basis, i.e. if models are retrained overnight in production then replicate this in evaluation.
- Obtain market data for evaluation in addition to observed outcomes.
- Back test possible strategies using model predictions.
- Obtain advanced market data and back test implementation of 'trading' strategies, i.e what time to bet, liquidity movements etc.

### General
- Add tests for all code.