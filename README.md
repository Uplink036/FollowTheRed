# FollowTheRed
A Jetson-powered robotics project focused on dynamic object tracking. This project aims to make a JetBot detect and follow a moving red box in real-time, using a custom movement strategy.

## Algorithm

```mermaid
---
config:
  theme: redux
  layout: dagre
---
flowchart TD
    Input[Get input image] --> Choice
    Choice{Use Heuristic or NN Model?}
    Choice -->|Heuristic| Heuristic[Detect red region]
    Heuristic --> Direction[Determine Direction]
    Direction --> Speed[Determine forward speed]
    Speed --> TrainNN[Train NN]

    Choice -->|NN| NN[Determine direction and speed from NN]

    TrainNN --> Increment[Increment gamma]
    NN --> Increment

    Increment --> ApplyMovement[Apply movement to robot]
    ApplyMovement --> Input
```
