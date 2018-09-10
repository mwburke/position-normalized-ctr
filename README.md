# Position Normalized Click Through Rates

Implementation of analysis to calculate the estimated click-through rates for ads and the estimated effect of ad position separately using maximum likelihood estimation from the following paper:

[Chen, Ye and Tak W. Yan. _Position-normalized click prediction in search advertising._ KDD (2012).](https://dl.acm.org/citation.cfm?doid=2339530.2339654)

## Technical Overview

Assumptions:

1. Clicking an ad is independent of its position, given that it is physically examined
2. Examining an ad is independent of its content or relevance,
given its position.


Given the above, you can decompose the click through rate (CTR) as a probability of clicking given that the user examines the ad into two factors, ad relevance and position prior upon:

$ p(click | adquery, position) = p(click | exam, adquery) p(exam|position)$

### Formulation

For more details on the expectation-maximization formulation, either read the original paper or view a [summary in this repo](formulation.pdf)

## Usage

### Data Format

Data should be a pandas Dataframe with the number of impressions and number of clicks for every combination of query-ad pair and position and organized into the following columns:

* `ad_query`: string or integer denoting unique ad/query combinations
* `position`: string or integer denoting unique ad position
* `impressions`: integer number of impressions
* `clicks`: integer number of clicks


### Run Instructions

You can either import the module and run the analysis on a dataframe to add the results to a script or you can run `pnctr.py <input_file.csv>`


## Dependencies

* pandas
* numpy
