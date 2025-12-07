# Simplified ActionPlanner Usage

The ActionPlanner has been simplified to use a **deterministic workflow** approach. No more complex multi-step reasoning - just clean, predictable paths based on captcha type.

## Core Concept

1. **Classify** the captcha type (checkbox, image_selection, drag_puzzle, text)
2. **Execute** deterministic workflow using only 2 tools:
   - `detect(prompt)` - Find objects in image (returns bounding boxes)
   - `point(prompt)` - Find a specific location (returns x, y coordinates)

## Workflows by Captcha Type

### 1. Checkbox
**No planning needed** - just call the tool directly:

```python
from src.planner import ActionPlanner

planner = ActionPlanner(backend="gemini")

# No need to classify - just find the checkbox
# In your solver:
# location = point_tool("checkbox center")
# click(location)
```

### 2. Image Selection
Deterministic path:
1. Classify → get instruction
2. Get detection target
3. Detect all instances
4. Click each one

```python
planner = ActionPlanner(backend="gemini")

# Step 1: Classify
classification = planner.classify("captcha.png")
# Returns: {"type": "image_selection", "instruction": "Select all traffic lights", "reasoning": "..."}

# Step 2: Get what to detect
target = planner.get_detection_target(
    instruction=classification["instruction"],
    image_path="captcha.png"
)
# Returns: "traffic light"

# Step 3 & 4: In your solver
# boxes = detect_tool(target)  # Returns [(x1, y1, x2, y2), ...]
# for box in boxes:
#     center = get_center(box)
#     click(center)
```

### 3. Drag Puzzle
Deterministic path:
1. Classify → get instruction
2. Get drag prompts (what to drag, where to drag)
3. Point to find source
4. Point to find destination
5. Drag from source to destination

```python
planner = ActionPlanner(backend="gemini")

# Step 1: Classify
classification = planner.classify("captcha.png")
# Returns: {"type": "drag_puzzle", "instruction": "Drag the piece to complete the puzzle", ...}

# Step 2: Get prompts for point() tool
prompts = planner.get_drag_prompts(
    instruction=classification["instruction"],
    image_path="captcha.png"
)
# Returns: {"draggable_prompt": "puzzle piece", "destination_prompt": "empty slot"}

# Step 3-5: In your solver
# source = point_tool(prompts["draggable_prompt"])
# destination = point_tool(prompts["destination_prompt"])
# drag(source, destination)
```

### 4. Text Captcha
Deterministic path:
1. Classify
2. Read text
3. Type it

```python
planner = ActionPlanner(backend="gemini")

# Step 1: Classify
classification = planner.classify("captcha.png")
# Returns: {"type": "text", "instruction": null, ...}

# Step 2: Read text
text = planner.read_text("captcha.png")
# Returns: "XyZ123"

# Step 3: In your solver
# type_text(text)
```

## API Reference

### ActionPlanner

```python
planner = ActionPlanner(
    backend="gemini",  # or "ollama", "openai", "deepseek"
    model=None,  # Auto-selected based on backend
    gemini_api_key=None,  # Or set GEMINI_API_KEY env var
)
```

### Methods

#### `classify(image_path: str) -> dict`
Classify the captcha type.

**Returns:**
```python
{
    "type": "checkbox" | "image_selection" | "drag_puzzle" | "text",
    "instruction": "instruction text or None",
    "reasoning": "brief explanation"
}
```

#### `get_detection_target(instruction: str, image_path: str) -> str`
For image_selection captchas, get the object class to detect.

**Example:** "Select all traffic lights" → "traffic light"

#### `get_drag_prompts(instruction: str, image_path: str) -> dict`
For drag_puzzle captchas, get prompts for the point() tool.

**Returns:**
```python
{
    "draggable_prompt": "what to drag",
    "destination_prompt": "where to drag to"
}
```

#### `read_text(image_path: str) -> str`
For text captchas, read the distorted text.

**Returns:** The text string to type

## Full Example

```python
from src.planner import ActionPlanner

def solve_captcha(image_path: str):
    planner = ActionPlanner(backend="gemini")
    
    # Step 1: Classify
    classification = planner.classify(image_path)
    captcha_type = classification["type"]
    instruction = classification["instruction"]
    
    # Step 2: Execute deterministic workflow
    if captcha_type == "checkbox":
        # Direct approach - no planner needed
        location = point("checkbox center")
        click(location)
        
    elif captcha_type == "image_selection":
        # Get what to detect
        target = planner.get_detection_target(instruction, image_path)
        
        # Detect and click all
        boxes = detect(target)
        for box in boxes:
            click(get_center(box))
        
    elif captcha_type == "drag_puzzle":
        # Get drag prompts
        prompts = planner.get_drag_prompts(instruction, image_path)
        
        # Find and drag
        source = point(prompts["draggable_prompt"])
        destination = point(prompts["destination_prompt"])
        drag(source, destination)
        
    elif captcha_type == "text":
        # Read and type
        text = planner.read_text(image_path)
        type_text(text)
```

## Benefits

1. **Predictable** - Same input = same workflow path
2. **Debuggable** - Easy to see what went wrong at each step
3. **Testable** - Each method can be tested independently
4. **Simple** - No complex multi-step reasoning or self-questioning
5. **Fast** - Minimal LLM calls needed

## Migration from Old Planner

Old way (complex):
```python
action = planner.plan(image_path, context, elements, prompt_text)
# Returns complex PlannedAction with many optional fields
```

New way (simple):
```python
# Step 1: Classify
classification = planner.classify(image_path)

# Step 2-N: Deterministic workflow based on type
if classification["type"] == "checkbox":
    # point("checkbox center") → click
    ...
```

