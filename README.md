# ReviewsMining
The project hopes to achieve data mining of product reviews through a multi-level purchase motivation understanding model based on natural language processing, so as to understand users' consumption motivation.
The project borrows the concept of fault tree from the field of mechanical engineering and innovates it by transferring it to the field of data mining to explore a multi-level understanding of consumer motivation.

# Requirements


`pandas >= 1.3.4`
`numpy >= 1.21.4`
`re`
`sys`
`jieba>=0.42.1`
`matplotlib>=3.5.0`
`sklearn>=0.0.post2`
`nltk>=3.8.1`


# Download
```
git clone https://github.com/ReviewsMiningbyTsinghua/ReviewsMining
cd ReviewsMining
```
# Usage
`$ python main.py`
# Basic principles
The project first uses the LDA method to extract all the review data by topic and build the body of the fault tree. Then, the review data is divided into words and classified according to the word nature. Different words are clustered separately to obtain the keywords of the word. The TFIDF method was used for clustering, and the perplexity was calculated to determine the optimal number of clusters. At this point, the keywords of each lexical nature can be obtained, and the trunk and leaves of the fault tree can be constructed according to their interconnection and matching relationship, and the information of the consumption scenario reflected by the review data can be mined after the fault tree is obtained.
After getting the keywords of each lexical nature, the mutual information between the keywords is calculated. Weights are added to the connected edges of the fault tree so as to visually discover the main consumption concerns and pain points of consumers in the consumption process.

# Summary
Using this project can help platforms and merchants discover diverse consumption scenarios more easily, drive consumption with scenarios, and at the same time deeply understand the strengths and weaknesses of products in order to make targeted improvements.
