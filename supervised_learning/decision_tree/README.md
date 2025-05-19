This project aims to implement decision trees from scratch. It is important for engineers to understand how the tools we use are built for two reasons. First, it gives us confidence in our skills. Second, it helps us when we need to build our own tools to solve unsolved problems.
The first three references point to historical papers where the concepts were first studied.
References 4 to 9 can help if you feel you need some more explanation about the way we split nodes.
William A. Belson is usually credited for the invention of decision trees (read reference 11).
Despite our efforts to make it efficient, we cannot compete with Sklearn’s implementations (since they are done in C). In real life, it is thus recommended to use Sklearn’s tools.
In this regard, it is warmly recommended to watch the video referenced as (10) above. It shows how to use Sklearn’s decision trees and insists on the methodology.

Once built, decision trees are binary trees : a node either is a leaf or has two children. It never happens that a node for which is_leaf is False has its left_child or right_child left unspecified.
The first three tasks are a warm-up designed to review the basics of class inheritance and recursion (nevertheless, the functions coded in these tasks will be reused in the rest of the project).
Our first objective will be to write a Decision_Tree.predict method that takes the explanatory features of a set of individuals and returns the predicted target value for these individuals.
Then we will write a method Decision_Tree.fit that takes the explanatory features and the targets of a set of individuals, and grows the tree from the root to the leaves to make it in an efficient prediction tool.
Once these tasks will be accomplished, we will introduce a new class Random_Forest that will also be a powerful prediction tool.
Finally, we will write a variation on Random_Forest, called Isolation_Random_forest, that will be a tool to detect outliers.

