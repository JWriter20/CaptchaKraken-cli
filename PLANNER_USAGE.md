# Simplified ActionPlanner Usage

The ActionPlanner has been simplified to use a **deterministic workflow** approach. No more complex multi-step reasoning - just clean, predictable paths based on captcha type.

1. Detect if the captcha is a grid or checkbox captcha:
    a. If it is a checkbox captcha use the find_checkbox tool to find the checkbox and return a click action with this bounding box.

    b. If it is a grid captcha, we detect the grid and apply the numbering and grid overlay, then we should just pass in the image to our LLM, and return the cells to click while filtering out already selected cells and loading cells (use our openCV-based selected cell detection tool)

2. General solving:
    a. Now we need to let the LLM decide what to do depending on the type of captcha. There are a few main types we will support:
    
    - Text captchas (e.g. "type the text in the image")
    - Drag puzzles (e.g. "drag the puzzle piece to the correct position", "complete the image")
    - Clicking described objects (e.g. "click the two similar shapes")
    - Analyzing the connected lines (e.g. "which watermelon is connected to the bird?")

    Each of these should start off with a prompt like "Solve this model, using our tools and describe the tool calls. 
