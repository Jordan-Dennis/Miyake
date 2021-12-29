# TO DOs:
 - I need to plan the `mcmc` profiling
 - I need to plan the `bosh` profiling 
 - I need to organise my directory
 - I need to devlop a naming convention 
 - I need to fix the injection recovery python script.
 - I need to add a `readme.md` that contains
     - directory key: This will explain where everything is and how to access it. 
         - Nicely formatted map of the directory 
 - I need to make my repo public following the improvements

# Naming conventions

# File organisations
So there are multiple possibilities here. I could chose to do my usual `.`extension system. The advantages are it is easy to find the files that you need. The dissadvantages arrise when loading the files from different directories. Another system would be logical divisions. By this I meen grouping files as `datasets`, `notebooks`, ect. This system and the one above could end up being very similar in structure. I think I prefer the second one a little more `pictures` could include any graphics but `.pdf` might be LaTex reports or `Gadfly` plots. A final one might be to have an environmental structure. `Julia ODE Suite Profiles` which contains all the files needed for that particular aspect in a single place and only those files. I could combine this with the second approach to yield a system that compartmentalised the directory first upon the purpose and then upon the files.  