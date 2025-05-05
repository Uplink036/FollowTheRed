# FollowTheRed
A Jetson-powered robotics project focused on dynamic object tracking. This project aims to make a JetBot detect and follow a moving red box in real-time, using a custom movement strategy.
``` mermaid
flowchart TD
    A --> B 
    B -- False --> C
    C[Algorithm Output] --> E
    E[Train Model] --> F
    B -- True --> D
    D[Model Output] --> F
    F[JetBot Moves]  

    A@{shape: circle, label: "Picture" }
    B@{shape: diamond, label:   "y < RANDOM" }
``` 
