
# Files with some LLM usage: 

## load_models.py: 
specific reasoning: I am rather horrible at using other packages. 
I acideally was blowing up my memory and overheating my computer.
LLms were used in these specific functions in this file. 
- _cuda_batch_size: figure out and create bounds for batching
- detect_device(): I was running on 2 different devices & needed help with specs that i did not want to read through. 
- mem_used(): model flagged which call to use per backend
-  load_model: was having issues with padding and not getting output 
- unload_model: I was having crazy memory blow up, helped with figuring out garbage collect. 


## analysis:
I used LLms for the syntax of color and plotting visualization aspects of analysis. 
I am not a pro on how to make 1 plot inch closer to another,
or the specific perticualities of seaborn vs other coloring packages.
- graphing dimensions: questions like, "I have 3 plots in one figure what is the command to make them closer to eachother?"
- graphing colors: was using a bad color default that was not ADA. Asked to change my current color scheme to another. 
- accidentally toggled off the axis and asked how to fix