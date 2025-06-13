# Voice-Guided Navigation System with YOLOv11
### AI-Powered Indoor Navigation for Enhanced Accessibility

---

## Project Overview

This project implements an innovative mobile application that provides voice navigation support for visually impaired individuals and visitors within complex indoor environments. The system integrates a YOLOv11 object detection model trained on a custom dataset with a Swift-based iOS application that determines user location and provides real-time voice-guided navigation.

## Key Innovation

Rather than attempting to recognize all objects within a building, our approach focuses on identifying **strategically selected unique objects** in each corridor to determine the user's position and orientation. This significantly reduces computational complexity while maintaining high navigation accuracy.

## System Architecture

### Core Components

1. **YOLOv11 Object Detection Model**
   - Custom-trained on unique corridor objects
   - Optimized for mobile deployment via CoreML
   - Real-time inference capabilities

2. **Directional Awareness Algorithm**
   - State-machine based navigation logic
   - Sequential corridor progression (1→2→3→4)
   - Object-based location confirmation

3. **iOS Application**
   - Swift-based implementation
   - Voice command interface
   - Real-time audio guidance
   - Gesture-based controls

## Technical Specifications

### Model Performance
- **Architecture**: YOLOv11m (modified for custom dataset)
- **mAP@0.5**: 96.7%
- **Precision**: 96.3%
- **Recall**: 93.9%
- **Input Size**: 640x640 pixels
- **Training**: 350 epochs on custom dataset

### Navigation Accuracy
- **Location Accuracy**: 94% (corridor identification)
- **Orientation Accuracy**: 92% (directional awareness)
- **Navigation Success Rate**: 85%
- **Average Time to Location Fix**: 1.5 seconds

## Dataset Information

- **Total Images**: 1,030
- **Annotations**: 1,332
- **Object Classes**: 8 unique objects
- **Train/Val/Test Split**: 70/20/10

### Object Classes and Performance

| Object Type | Test Accuracy | Validation Accuracy |
|-------------|---------------|---------------------|
| Fire Extinguisher | 96.0% | 99.0% |
| Fire Hose Cabinet | 96.0% | 100% |
| Human Painting | 100% | 100% |
| Main Door | 93.0% | 98.0% |
| Peacock | 100% | 100% |
| Printer | 97.0% | 86.0% |
| Trash Bin | 100% | 93.0% |
| Vending Machine | 100% | 100% |

## Navigation Algorithm

The system implements a sequential navigation approach through four corridors:

```
START → Initialize Navigation → Capture Camera Frame → YOLO11 Detection
   ↓
Objects Detected? → YES → Process Detection & Announce Object
   ↓                 NO ↘
Corridor Identified? → YES → Confirm Corridor & Give Instructions
   ↓                    NO ↘
Turn Object Found? → YES → Transition to Next Corridor
   ↓                  NO ↘
Destination Reached? → YES → END
   ↓                    NO → Continue Loop
```

### Corridor Identification Logic

| Corridor | Identification Objects | Turn Trigger | Navigation Action |
|----------|----------------------|--------------|-------------------|
| Corridor 1 | Fire Hose Cabinet + Vending Machine | Printer | Turn left → Corridor 2 |
| Corridor 2 | Fire Extinguisher | End of corridor | Turn left → Corridor 3 |
| Corridor 3 | Peacock + Human Painting | Main Door | Turn left → Corridor 4 |
| Corridor 4 | Trash Bin | - | Destination reached |

## Implementation Details

### Technical Stack
- **Language**: Swift 5.7
- **Framework**: iOS UIKit
- **ML Framework**: CoreML, Vision
- **Development Environment**: Xcode 14.2
- **Deployment Target**: iOS 15.0+

### Key Features
- Real-time object detection with bounding box visualization
- Voice synthesis for navigation instructions
- Gesture controls for status updates
- Stable detection threshold (3 consecutive frames)
- Transition state management

### Model Conversion
```python
from ultralytics import YOLO
model = YOLO("best.pt") 
model.export(format="coreml", int8=True, nms=True, imgsz=[640, 384])
```

## System Features

1. **Accessibility First**
   - Voice-guided navigation instructions
   - Simple gesture controls
   - Clear audio feedback
   - No external infrastructure required

2. **Robust Detection**
   - Multiple angle training data
   - Lighting variation handling
   - Occlusion resistance
   - Historical tracking during object absence

3. **User Experience**
   - Natural language instructions
   - Current location on-demand
   - Automatic reorientation
   - Battery-efficient operation

## Testing & Evaluation

### Performance Metrics
- Object Recognition Tests: 94.1% mAP
- Location Awareness: 94% accuracy
- Navigation Success: 85% completion rate
- User Satisfaction: 4.0/5 rating

### Real-World Testing
- 20 participants (5 visually impaired, 15 sighted)
- Multiple lighting conditions
- Various crowd densities
- Different times of day

## Future Enhancements

1. **Coverage Expansion**
   - Multi-floor navigation
   - Stairwell and elevator integration
   - Outdoor-indoor transitions

2. **Algorithm Improvements**
   - IMU sensor fusion
   - Crowd-sourced object updates
   - Personalized navigation preferences

3. **Optimization**
   - Battery consumption reduction
   - Edge computing enhancements
   - Offline capability expansion



## Contributing

This project was developed as an innovative solution for indoor navigation accessibility. Contributions focusing on improving accuracy, expanding coverage, or enhancing user experience are welcome.

## License

This project is licensed under the AGPL-3.0 License.

## Acknowledgments

- Developed at Sabancı University
- Built with Ultralytics YOLOv11 framework

---

*Enhancing indoor accessibility through AI-powered navigation technology*
