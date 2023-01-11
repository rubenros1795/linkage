# Observations

## Linkage trend

_Means_
- Sharp increase in the first years of the timeframe (1946-1948)
- After 1948, a period of relative stability follows, until 1957. Some minor bumps are visible (for example a decline between 1948-1950)
- Between 1957 and 1961, a steep increase sets in. Events from parliamentary and political history that are associated with this period are:
    - the installment of permanent committees in the Lower House
    - the establishment of the European Economic Community
- After 1961, the rising trend flattens and remains stable until the end of the time period

_Standard Deviations_
- After a minor decline in 1946-1947, the standard deviations rise rapidly from 1947 until 1949. 
- From 1949 until 1962, there appears to be some seasonality in the standard deviation trend. This seems unrelated to both parliamentary cycles and cabinet changes. 
- After 1962, a strong decline in the standard deviation is visible.
- Halfway 1964, however, a strong increase follows the decline, a pattern that continues until the end of the time period.

## Quantitative Explanations
The trends visible in the linkage means and standard deviations are not easy to explain based on known historical events or historiography. Therefore, we harness several other quantitative tools to contextualize the trends.


- Shannon Entropy:
    - We calculate the shannon entropy over the daily mean topic distributions
    - We extract a signal using non-linear adaptive filtering
    - Overall, there is not much variation in entropy, values range from 7.85 to 7.9.
    - Still, there are some trends that catch the eye:
        - Between 1945 and 1948, the entropy increases significantly. Smaller entropy values could be an indicator of a skewed distribution of topics, i.e. some topics occur abdunantly in a text, while most others hardly occur. Until 1948, this skewedness rapidly decreases. This is probably explained by the parliamentary processing of wartime issues that capture most of the parliamentary attention, but gradually fade from the proceedings as the distance to the war grew.
        - After a period of stability between 1948 and 1958, the entropy slowly declines, with an acceleration of this decline after 1966. 
- Linkage Pair Correlation:
    - We calculate the correlation (np.correlate) between the daily flattened linkage vector, and the _N_ (50 in the current setup) previous daily linkage vectors and average them.
    - The results is a time series that shows the novelty of the linkage vectors over time.
    - The time series shows some interesting trends:
        - from 1945 until 1950, the correlation increases evenly.
        - After a period of stability, the correlation increases drastically in 1958
        - In 1962, it declines, followed by an increase from 1966 onwards.
    - The correlation plot shows some similarities with the linkage mean and standard deviation.
    - Especially the years 1957/1958 show up again as turning points. 
- Linkage Pair Novelty & Resonance (to be improved)
    - Novelty scores rapidly increase in the period 1945-1948, in line with the decreasing linkage and increasing entropy
    - Resonance scores show no overall trend, but several peaks. 
- Metadata Features: Government Presence
    - The proportion of government members present in debates steeply drops after 1964. If this is linked to linkage at all, this would mean that linkage, after 1964 is something caused by members speaking, instead of government members.

- Individual Linkage Pair Trends (working on this)
    - Comparing the mean linkage with individual linkage pair time series could help in identifying concrete topics that contribute to the patterns described.
    - Calculating correlations between mean linkage and individual pairs shows some policy domains that are relevant, especially to the bump in 1957/1958:
        - Macro-Economic Affairs
        - European Integration