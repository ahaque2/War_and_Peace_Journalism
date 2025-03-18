# Analyzing War News Using War and Peace Journalism Framework

**Abstract.** We apply the war and peace journalism framework to examine news coverage of the ongoing Gaza conflict, analyzing three major mainstream news publishers -- Fox News (US), BBC (UK), and Aljazeera (Qatar).
We develop a crowdsourced annotated dataset of news headlines, including labels for war and peace frames, as well as role labels identifying the portrayal of conflict entities as victims or villains. 
We use this annotated dataset to fine-tune pretrained language models to conduct the analysis.
Our findings reveal significant differences in coverage across publishers, particularly in the portrayal of victims and villains. 
Fox News frames Hamas as a villain and Israel as a victim, whereas Aljazeera portrays Israel as a villain and Palestine as a victim. 
In contrast, BBC provides a more balanced coverage.
Although publishers differ significantly in their use of war and peace frames, the overall picture remains inconclusive since all publishers use both frames. 
Additionally, we find that the presence of a peace frame does not guarantee peace-supportive coverage, and additional context is needed to make that determination. 
Moreover, beyond frame selection, the presentation of news itself plays a crucial role in shaping interpretation.
Finally, we highlight key challenges and limitations of the war and peace journalism framework in practice, offering insights for future research.


## Running the code

- *1_Benchmark_dataset.ipynb* code to benchmark language models on the annotated war and peace journalism dataset.

- *2_BERT_with_custom_features.ipynb* code that uses custom features like POS (part-of-speech), named entities, and transformation in a moral embedding vector space.

- *3_Classify_News.ipynb* code to compute scores for war and peace frames and victim and villain portrayal for each headline.

- *4_news_results_analysis.ipynb* code for news analysis

- *5_statistical_tests.ipynb* code for statistical tests

- *6_visualizations.ipynb* code for visualization

Note: Minor changes may be required to run the code in some cases. Most of these changes are usually present as comments and just need to be uncommented. The changes are needed as there are several class (hence, different class names and file), and each class has different labels (i.e., different column names). Only these minor adjustments are needed to run the code for different classes. 