//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  Main View Controller for Ultralytics YOLO App
//  Enhanced with Sequential Corridor Navigation System (1→2→3→4)

import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision

// Navigation state for tracking current phase of navigation
enum NavigationState {
    case initialState        // Starting state, no corridor determined yet
    case detectingCorridor   // Looking for corridor identification objects
    case corridorConfirmed   // Corridor confirmed, ready for next instruction
    case walkingToTurnObject // Walking to find turn object
    case readyToTurn         // Turn object found, ready to turn
    case turningToCorridor   // Currently turning to next corridor
    case destinationReached  // Final destination reached
}

// Detected object types with English naming
enum DetectedObjectType: String, CaseIterable {
    case fireExtinguisher = "fire extinguisher"
    case fireHoseCabinet = "fire hose cabinet"
    case humanPainting = "human painting"
    case mainDoor = "main door"
    case peacock = "peacock"
    case printer = "printer"
    case trashBin = "trash bin"
    case vendingMachine = "vending machine"
    case unknown = "unknown"
}

// Corridor definitions with new sequential navigation logic
enum Corridor: Int, CaseIterable {
    case corridor1 = 1
    case corridor2 = 2
    case corridor3 = 3
    case corridor4 = 4
    
    // Corridor identification objects (must see ALL for confirmation)
    var identificationObjects: [DetectedObjectType] {
        switch self {
        case .corridor1:
            return [.fireHoseCabinet, .vendingMachine] // Both required
        case .corridor2:
            return [.fireExtinguisher] // Single object
        case .corridor3:
            return [.peacock, .humanPainting] // Both required
        case .corridor4:
            return [.trashBin] // Single object (only if no other identifying objects present)
        }
    }
    
    // Turn objects for leaving this corridor
    var turnObject: DetectedObjectType? {
        switch self {
        case .corridor1:
            return .printer
        case .corridor2:
            return nil // Turn at end of corridor
        case .corridor3:
            return .mainDoor
        case .corridor4:
            return nil // Destination reached
        }
    }
    
    // Next corridor after turning
    var nextCorridor: Corridor? {
        switch self {
        case .corridor1:
            return .corridor2
        case .corridor2:
            return .corridor3
        case .corridor3:
            return .corridor4
        case .corridor4:
            return nil // Final destination
        }
    }
    
    // Corridor name in English
    var name: String {
        return "Corridor \(rawValue)"
    }
}

var mlModel = try! yolo11m(configuration: mlmodelConfig).model
var mlmodelConfig: MLModelConfiguration = {
  let config = MLModelConfiguration()

  if #available(iOS 17.0, *) {
    config.setValue(1, forKey: "experimentalMLE5EngineUsage")
  }

  return config
}()

class ViewController: UIViewController {
    
    // MARK: - Navigation Properties
    private var navigationState: NavigationState = .initialState // Start with initial state
    private var currentCorridor: Corridor? = nil // Start with no corridor determined
    private var detectedObjects: Set<DetectedObjectType> = []
    private var confirmedCorridorObjects: Set<DetectedObjectType> = [] // Track objects already identified in current corridor
    private var recentDetections: [DetectedObjectType: Int] = [:] // Track consecutive detections
    private var lastNavigationAnnouncement = Date()
    private var lastObjectAnnouncement = Date()
    private let navigationCooldown: TimeInterval = 3.0
    private let objectAnnouncementCooldown: TimeInterval = 1.5
    private var announcedObjects: Set<DetectedObjectType> = []
    private var stableDetectionThreshold = 3 // Require multiple detections to confirm
    private var isProcessingDetection = false // Flag to prevent multiple simultaneous detections
    private var corridorIdentified = false // Flag to indicate if corridor identification is complete
    private var humanPaintingDetected = false // Flag to track if human painting is detected in corridor 3
    private var peacockDetected = false // Flag to track if peacock is detected in corridor 3
    private var isTransitioning = false // Flag to prevent object detection during corridor transitions
    
    // MARK: - UI Outlets
    @IBOutlet var videoPreview: UIView!
    @IBOutlet var View0: UIView!
    @IBOutlet var segmentedControl: UISegmentedControl!
    @IBOutlet var playButtonOutlet: UIBarButtonItem!
    @IBOutlet var pauseButtonOutlet: UIBarButtonItem!
    @IBOutlet var slider: UISlider!
    @IBOutlet var sliderConf: UISlider!
    @IBOutlet weak var sliderConfLandScape: UISlider!
    @IBOutlet var sliderIoU: UISlider!
    @IBOutlet weak var sliderIoULandScape: UISlider!
    @IBOutlet weak var labelName: UILabel!
    @IBOutlet weak var labelFPS: UILabel!
    @IBOutlet weak var labelZoom: UILabel!
    @IBOutlet weak var labelVersion: UILabel!
    @IBOutlet weak var labelSlider: UILabel!
    @IBOutlet weak var labelSliderConf: UILabel!
    @IBOutlet weak var labelSliderConfLandScape: UILabel!
    @IBOutlet weak var labelSliderIoU: UILabel!
    @IBOutlet weak var labelSliderIoULandScape: UILabel!
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var focus: UIImageView!
    @IBOutlet weak var toolBar: UIToolbar!
    
    // MARK: - Core Properties
    let speechSynthesizer = AVSpeechSynthesizer()
    var latestDetectionKeyword: String?
    var isSpeaking: Bool = false
    
    // Zoom variables
    var minimumZoom: CGFloat = 1.0
    var maximumZoom: CGFloat = 10.0
    var lastZoomFactor: CGFloat = 1.0
    
    let selection = UISelectionFeedbackGenerator()
    var detector = try! VNCoreMLModel(for: mlModel)
    var session: AVCaptureSession!
    var videoCapture: VideoCapture!
    var currentBuffer: CVPixelBuffer?
    var framesDone = 0
    var t0 = 0.0  // inference start
    var t1 = 0.0  // inference dt
    var t2 = 0.0  // inference dt smoothed
    var t3 = CACurrentMediaTime()  // FPS start
    var t4 = 0.0  // FPS dt smoothed
    var longSide: CGFloat = 3
    var shortSide: CGFloat = 4
    var frameSizeCaptured = false
    
    // Developer mode
    let developerMode = UserDefaults.standard.bool(forKey: "developer_mode")
    let save_detections = false
    let save_frames = false
    
    // Bounding box related
    let maxBoundingBoxViews = 100
    var boundingBoxViews = [BoundingBoxView]()
    var colors: [String: UIColor] = [:]
    let ultralyticsColorsolors: [UIColor] = [
        UIColor(red: 4 / 255, green: 42 / 255, blue: 255 / 255, alpha: 0.6),  // #042AFF
        UIColor(red: 11 / 255, green: 219 / 255, blue: 235 / 255, alpha: 0.6),  // #0BDBEB
        UIColor(red: 243 / 255, green: 243 / 255, blue: 243 / 255, alpha: 0.6),  // #F3F3F3
        UIColor(red: 0 / 255, green: 223 / 255, blue: 183 / 255, alpha: 0.6),  // #00DFB7
        UIColor(red: 17 / 255, green: 31 / 255, blue: 104 / 255, alpha: 0.6),  // #111F68
        UIColor(red: 255 / 255, green: 111 / 255, blue: 221 / 255, alpha: 0.6),  // #FF6FDD
        UIColor(red: 255 / 255, green: 68 / 255, blue: 79 / 255, alpha: 0.6),  // #FF444F
        UIColor(red: 204 / 255, green: 237 / 255, blue: 0 / 255, alpha: 0.6),  // #CCED00
        UIColor(red: 0 / 255, green: 243 / 255, blue: 68 / 255, alpha: 0.6),  // #00F344
        UIColor(red: 189 / 255, green: 0 / 255, blue: 255 / 255, alpha: 0.6),  // #BD00FF
        UIColor(red: 0 / 255, green: 180 / 255, blue: 255 / 255, alpha: 0.6),  // #00B4FF
        UIColor(red: 221 / 255, green: 0 / 255, blue: 186 / 255, alpha: 0.6),  // #DD00BA
        UIColor(red: 0 / 255, green: 255 / 255, blue: 255 / 255, alpha: 0.6),  // #00FFFF
        UIColor(red: 38 / 255, green: 192 / 255, blue: 0 / 255, alpha: 0.6),  // #26C000
        UIColor(red: 1 / 255, green: 255 / 255, blue: 179 / 255, alpha: 0.6),  // #01FFB3
        UIColor(red: 125 / 255, green: 36 / 255, blue: 255 / 255, alpha: 0.6),  // #7D24FF
        UIColor(red: 123 / 255, green: 0 / 255, blue: 104 / 255, alpha: 0.6),  // #7B0068
        UIColor(red: 255 / 255, green: 27 / 255, blue: 108 / 255, alpha: 0.6),  // #FF1B6C
        UIColor(red: 252 / 255, green: 109 / 255, blue: 47 / 255, alpha: 0.6),  // #FC6D2F
        UIColor(red: 162 / 255, green: 255 / 255, blue: 11 / 255, alpha: 0.6),  // #A2FF0B
    ]
    
    lazy var visionRequest: VNCoreMLRequest = {
        let request = VNCoreMLRequest(
            model: detector,
            completionHandler: {
                [weak self] request, error in
                self?.processObservations(for: request, error: error)
            })
        request.imageCropAndScaleOption = .scaleFill
        return request
    }()

    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        slider.value = 30
        setLabels()
        setUpBoundingBoxViews()
        setUpOrientationChangeNotification()
        startVideo()
        
        // Setup gesture recognizers - only keeping single and double tap
        setupGestureRecognizers()
        
        // Speech synthesizer delegate
        speechSynthesizer.delegate = self
        
        // Start navigation system automatically
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            self.startNavigation()
        }
    }
    
    private func setupGestureRecognizers() {
        // Single tap - current detection info
        let tapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        self.view.addGestureRecognizer(tapGestureRecognizer)
        
        // Double tap - current status and corridor info
        let doubleTapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(handleDoubleTap(_:)))
        doubleTapGestureRecognizer.numberOfTapsRequired = 2
        self.view.addGestureRecognizer(doubleTapGestureRecognizer)
        
        // Setup tap dependencies
        tapGestureRecognizer.require(toFail: doubleTapGestureRecognizer)
    }
    
    // MARK: - Gesture Handlers
    @objc func handleTap(_ sender: UITapGestureRecognizer) {
        print("Single tap detected")
        
        var message = ""
        if let keyword = latestDetectionKeyword {
            message = "Last detection: \(keyword). "
        } else {
            message = "No object detected yet. "
        }
        
        if let corridor = currentCorridor {
            message += "Currently in \(corridor.name). Navigation state: \(navigationState)."
        } else {
            message += "No corridor determined yet. Please scan for corridor identification objects."
        }
        
        speak(message)
    }
    
    @objc func handleDoubleTap(_ sender: UITapGestureRecognizer) {
        print("Double tap detected")
        
        var message = ""
        if let corridor = currentCorridor {
            message = "You are currently in \(corridor.name). "
            
            switch navigationState {
            case .initialState:
                message += "Navigation just started."
            case .detectingCorridor:
                message += "Scanning to detect corridor objects."
            case .corridorConfirmed:
                message += "Corridor confirmed. Ready for next instruction."
            case .walkingToTurnObject:
                message += "Walk and scan to find the turning object."
            case .readyToTurn:
                message += "Ready to turn."
            case .turningToCorridor:
                message += "Turning to next corridor."
            case .destinationReached:
                message += "Destination reached. Kamer teacher's room is on your right."
            }
        } else {
            message = "No corridor determined yet. Please scan for corridor identification objects."
        }
        
        speak(message)
    }

    private func startNavigation() {
        navigationState = .initialState
        currentCorridor = nil
        detectedObjects.removeAll()
        recentDetections.removeAll()
        announcedObjects.removeAll()
        confirmedCorridorObjects.removeAll()
        corridorIdentified = false
        humanPaintingDetected = false
        peacockDetected = false
        isTransitioning = false

        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            self.speak("Navigation has begun. Your destination classroom has been registered. To compute the optimal path to professor Kamer's office, scan nearby objects around you.")
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 4.0) {
            self.navigationState = .detectingCorridor
        }
    }

    
    // FIXED: processDetection function with transition blocking
    private func processDetection(_ objectType: DetectedObjectType, confidence: Float) {
        // Skip if we're transitioning between corridors
        if isTransitioning {
            return
        }
        
        // Skip if we're already processing
        if isProcessingDetection {
            return
        }
        
        isProcessingDetection = true
        
        // Update detection count for this object
        recentDetections[objectType] = (recentDetections[objectType] ?? 0) + 1
        
        // Decay other detections to prevent false positives
        for key in recentDetections.keys {
            if key != objectType {
                recentDetections[key] = max((recentDetections[key] ?? 0) - 1, 0)
                if recentDetections[key] == 0 {
                    recentDetections.removeValue(forKey: key)
                }
            }
        }
        
        // Only process stable detections (seen multiple times)
        if recentDetections[objectType] ?? 0 >= stableDetectionThreshold {
            // Add to current detection set if confidence is high
            if confidence > 0.6 {
                // Only announce objects that are relevant to the current corridor and navigation state
                let shouldAnnounce = shouldAnnounceObject(objectType)
                
                if shouldAnnounce {
                    detectedObjects.insert(objectType)
                    
                    // Announce the object detection
                    announceObjectDetection(objectType)
                    
                    // Process based on current state
                    switch navigationState {
                    case .initialState:
                        // Wait for corridor detection
                        break
                        
                    case .detectingCorridor:
                        checkCorridorIdentification()
                        
                    case .corridorConfirmed:
                        break
                        
                    case .walkingToTurnObject:
                        checkForTurnObject()
                        
                    case .readyToTurn:
                        break
                        
                    case .turningToCorridor:
                        // Should not process during transition
                        break
                        
                    case .destinationReached:
                        break
                    }
                }
            }
        }
        
        isProcessingDetection = false
    }
    
    // FIXED: shouldAnnounceObject with better corridor filtering
    private func shouldAnnounceObject(_ objectType: DetectedObjectType) -> Bool {
        // Check if the object is already confirmed
        if confirmedCorridorObjects.contains(objectType) {
            return false
        }
        
        // During transitions, don't announce anything
        if isTransitioning {
            return false
        }
        
        // Trash bin should only be announced in corridor 4 detection phase
        if objectType == .trashBin {
            return currentCorridor == .corridor4 && navigationState == .detectingCorridor
        }
        
        // In corridor 1, only announce fire hose cabinet, vending machine, and printer
        if currentCorridor == .corridor1 {
            if corridorIdentified {
                return objectType == .printer
            } else {
                return objectType == .fireHoseCabinet || objectType == .vendingMachine
            }
        }
        
        // In corridor 2, only announce fire extinguisher
        if currentCorridor == .corridor2 {
            return objectType == .fireExtinguisher && navigationState == .detectingCorridor
        }
        
        // In corridor 3, announce peacock, human painting, and main door
        if currentCorridor == .corridor3 {
            if navigationState == .detectingCorridor {
                return objectType == .peacock || objectType == .humanPainting
            } else if navigationState == .walkingToTurnObject {
                return objectType == .mainDoor
            }
        }
        
        // In corridor 4, only announce trash bin
        if currentCorridor == .corridor4 {
            return objectType == .trashBin && navigationState == .detectingCorridor
        }
        
        // If no corridor identified yet, announce identification objects for corridor 1
        if currentCorridor == nil && navigationState == .detectingCorridor {
            return objectType == .fireHoseCabinet || objectType == .vendingMachine
        }
        
        return false
    }
    
    // FIXED: announceObjectDetection with corridor 3 logic
    private func announceObjectDetection(_ objectType: DetectedObjectType) {
        let now = Date()
        if now.timeIntervalSince(lastObjectAnnouncement) < objectAnnouncementCooldown {
            return
        }
        if confirmedCorridorObjects.contains(objectType) {
            return
        }
        if !shouldAnnounceObject(objectType) {
            return
        }

        // FIRE HOSE CABINET için özel anons (Corridor 1 identification phase)
        if objectType == .fireHoseCabinet
            && (currentCorridor == nil || currentCorridor == .corridor1)
            && navigationState == .detectingCorridor
        {
            speak("Keep the fire hose cabinet on your right and walk forward, scanning around until you see the vending machine.")
            confirmedCorridorObjects.insert(.fireHoseCabinet)
            lastObjectAnnouncement = now
            announcedObjects.insert(objectType)
            return
        }

        // Corridor 3 için özel logic
        if currentCorridor == .corridor3 && navigationState == .detectingCorridor {
            if objectType == .peacock {
                speak("Peacock detected. Corridor 3 confirmed. Human painting upfront scan until you find human painting.")
                peacockDetected = true
            } else if objectType == .humanPainting {
                speak("Human painting detected.")
                humanPaintingDetected = true
            }
            // İkisi de algılandıysa corridor 3'ü onayla
            if peacockDetected && humanPaintingDetected {
                confirmCorridor(.corridor3)
            }
        } else if currentCorridor == .corridor3 && navigationState == .walkingToTurnObject && objectType == .mainDoor {
            // main door algılandı, corridor 3 için turn object
            return // checkForTurnObject fonksiyonunda ele alınıyor
        } else {
            // Diğer nesneler için standart anons
            speak("\(objectType.rawValue) detected")
        }

        // Son durum güncellemeleri
        confirmedCorridorObjects.insert(objectType)
        lastObjectAnnouncement = now
        announcedObjects.insert(objectType)
    }

    
    // FIXED: checkCorridorIdentification
    private func checkCorridorIdentification() {
        // Skip if not in detecting state
        guard navigationState == .detectingCorridor else { return }
        
        // If corridor is already identified, skip
        if corridorIdentified && currentCorridor != nil {
            return
        }
        
        // Special case for Corridor 1 - must have BOTH fire hose cabinet AND vending machine
        if currentCorridor == nil && detectedObjects.contains(.fireHoseCabinet) && detectedObjects.contains(.vendingMachine) {
            confirmCorridor(.corridor1)
            return
        }
        
        // Corridor 2 - currentCorridor should be .corridor2
        if currentCorridor == .corridor2 && detectedObjects.contains(.fireExtinguisher) {
            confirmCorridor(.corridor2)
            return
        }
        
        // Corridor 3 - must have BOTH peacock AND human painting
        if currentCorridor == .corridor3 && peacockDetected && humanPaintingDetected {
            confirmCorridor(.corridor3)
            return
        }
        
        // Corridor 4 - currentCorridor should be .corridor4
        if currentCorridor == .corridor4 && detectedObjects.contains(.trashBin) {
            confirmCorridor(.corridor4)
            return
        }
    }

    // FIXED: confirmCorridor with proper timing
    private func confirmCorridor(_ corridor: Corridor) {
        guard navigationState == .detectingCorridor else { return }
        
        // Mark corridor as identified
        corridorIdentified = true
        navigationState = .corridorConfirmed
        
        // Mark corridor identification objects as confirmed
        for object in corridor.identificationObjects {
            if detectedObjects.contains(object) {
                confirmedCorridorObjects.insert(object)
            }
        }
        
        // Specific announcements for each corridor confirmation
        switch corridor {
        case .corridor1:
            currentCorridor = corridor
            speak("Corridor 1 detected. Objects found: fire hose cabinet and vending machine. Walk and scan objects until you find the printer to turn left. Scan for fire extinguisher for confirmation.")
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                self.navigationState = .walkingToTurnObject
            }
            
        case .corridor2:
            speak("Fire extinguisher detected. Corridor 2 confirmed. Keep the fire extinguisher on your right and walk straight until the end of the corridor.")
            
            // Give more time for walking (10 seconds)
            DispatchQueue.main.asyncAfter(deadline: .now() + 10.0) {
                self.navigationState = .readyToTurn
                self.speak("Turn left in 4 seconds.")
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    self.executeTransitionToNextCorridor()
                }
            }
            
        case .corridor3:
            speak("Walk and scan to find the main door.")
            navigationState = .walkingToTurnObject
            
        case .corridor4:
            DispatchQueue.main.asyncAfter(deadline: .now() + 4.0) {
                self.speak("Trash bin detected. Corridor 4 confirmed. Kamer teacher's room is on your right.")
                self.navigationState = .destinationReached
            }

        }
        
        // Reset detection tracking for next phase
        detectedObjects.removeAll()
        recentDetections.removeAll()
    }
    
    private func checkForTurnObject() {
        guard let corridor = currentCorridor, navigationState == .walkingToTurnObject else { return }
        
        guard let turnObject = corridor.turnObject else {
            // No specific turn object (like corridor 2) - handled in confirmCorridor
            return
        }
        
        if detectedObjects.contains(turnObject) {
            navigationState = .readyToTurn
            
            switch corridor {
            case .corridor1:
                speak("Printer detected. Turn left now. You will be in the second corridor.")
                // Wait 2 seconds before announcing next corridor
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    self.executeTransitionToNextCorridor()
                }
                
            case .corridor3:
                speak("Main door detected. Turn left in 3 seconds.")
                // Wait 2 seconds before announcing next corridor
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    self.executeTransitionToNextCorridor()
                }
                
            default:
                speak("Turn left now.")
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    self.executeTransitionToNextCorridor()
                }
            }
            
            // Add turn object to confirmed objects
            confirmedCorridorObjects.insert(turnObject)
        }
    }
    
    // FIXED: executeTransitionToNextCorridor with transition flag
    private func executeTransitionToNextCorridor() {
        guard let corridor = currentCorridor, let nextCorridor = corridor.nextCorridor else {
            navigationState = .destinationReached
            speak("Destination reached. You are now in corridor 4. Kamer teacher's room is on your right.")
            return
        }
        
        // Set transition flag to prevent object detection during transition
        isTransitioning = true
        navigationState = .turningToCorridor
        
        // Clear all detection data
        detectedObjects.removeAll()
        recentDetections.removeAll()
        announcedObjects.removeAll()
        confirmedCorridorObjects.removeAll()
        corridorIdentified = false
        humanPaintingDetected = false
        peacockDetected = false
        
        // Update currentCorridor immediately
        currentCorridor = nextCorridor
        
        // Announce transition immediately
        switch nextCorridor {
        case .corridor2:
            speak("You are now in Corridor 2. Look for fire extinguisher for confirmation.")
        case .corridor3:
            speak("You are now in Corridor 3. Look for peacock and human painting for confirmation.")
        case .corridor4:
            speak("You are now in Corridor 4. Look for trash bin for confirmation.")
        default:
            speak("Entering \(nextCorridor.name).")
        }
        
        // Wait before allowing object detection again
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
            self.isTransitioning = false
            self.navigationState = .detectingCorridor
        }
    }
    
    private func checkCorridorTransition() {
        // This method can be used to detect when we've successfully moved to the next corridor
        // For now, we're using time-based transition in executeTransitionToNextCorridor
    }
    
    private func speak(_ text: String) {
        // Implement cooldown to prevent too frequent announcements for navigation instructions
        let now = Date()
        if text.contains("Corridor") || text.contains("Turn") || text.contains("Walk") {
            if now.timeIntervalSince(lastNavigationAnnouncement) < navigationCooldown {
                return
            }
            lastNavigationAnnouncement = now
        }
        
        print("🗣️ Speaking: \(text)")
        
        // Stop any current speech
        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
        
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US") // English voice
        utterance.rate = 0.5
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0
        
        // Ensure speech happens and log it
        DispatchQueue.main.async {
            self.speechSynthesizer.speak(utterance)
            print("Started speaking: \(text)")
        }
    }

    // MARK: - Speech Settings
    func showSpeechSettings() {
        let alertController = UIAlertController(
            title: "Speech Settings",
            message: "Customize speech announcements",
            preferredStyle: .actionSheet
        )
        
        // Speech rate options
        alertController.addAction(UIAlertAction(
            title: "Speech Rate",
            style: .default,
            handler: { _ in
                let speedAlert = UIAlertController(
                    title: "Speech Rate",
                    message: "Select speech rate",
                    preferredStyle: .actionSheet
                )
                
                let speeds = ["Very Slow", "Slow", "Normal", "Fast", "Very Fast"]
                let speedValues: [Float] = [0.3, 0.4, 0.5, 0.6, 0.7]
                
                for (index, speed) in speeds.enumerated() {
                    speedAlert.addAction(UIAlertAction(
                        title: speed,
                        style: .default,
                        handler: { _ in
                            // Test speech example
                            let utterance = AVSpeechUtterance(string: "This is a \(speed) speech rate example")
                            utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
                            utterance.rate = speedValues[index]
                            self.speechSynthesizer.speak(utterance)
                        }
                    ))
                }
                
                speedAlert.addAction(UIAlertAction(
                    title: "Cancel",
                    style: .cancel
                ))
                
                self.present(speedAlert, animated: true)
            }
        ))
        
        // Cancel button
        alertController.addAction(UIAlertAction(
            title: "Cancel",
            style: .cancel
        ))
        
        present(alertController, animated: true)
    }
    
    func showPredictions(predictions: [VNRecognizedObjectObservation]) {
            // Hide all bounding boxes first
            for box in self.boundingBoxViews {
                box.hide()
            }
            
            // Show detected objects
            for i in 0..<predictions.count {
                if i < self.boundingBoxViews.count {
                    let prediction = predictions[i]
                    
                    if let topLabelObservation = prediction.labels.first {
                        let className = topLabelObservation.identifier
                        let confidence = topLabelObservation.confidence
                        
                        self.boundingBoxViews[i].show(
                            frame: prediction.boundingBox,
                            label: "\(className) \(String(format: "%.2f", confidence))",
                            color: self.colors[className] ?? UIColor.red,
                            alpha: 0.6,
                            bounds: self.videoPreview.bounds
                        )
                    }
                }
            }
        }
        
        func processObservations(for request: VNRequest, error: Error?) {
            DispatchQueue.main.async {
                if let results = request.results as? [VNRecognizedObjectObservation] {
                    self.showPredictions(predictions: results)
                    
                    // Process high confidence detections for navigation
                    for result in results {
                        if let topLabel = result.labels.first {
                            let objectLabel = topLabel.identifier.lowercased()
                            let confidence = topLabel.confidence
                            
                            // Convert to DetectedObjectType and process for navigation
                            if let detectedType = DetectedObjectType(rawValue: objectLabel) {
                                self.processDetection(detectedType, confidence: confidence)
                                self.latestDetectionKeyword = objectLabel
                            }
                        }
                    }
                    
                    // Update latest detection for tap gesture
                    if let bestClass = results.first?.labels.first?.identifier {
                        self.latestDetectionKeyword = bestClass
                    }
                } else {
                    self.showPredictions(predictions: [])
                }
                
                // Measure FPS
                if self.t1 < 10.0 {  // valid dt
                    self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
                }
                self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
                self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)  // t2 seconds to ms
                self.t3 = CACurrentMediaTime()
            }
        }
        
        func predict(sampleBuffer: CMSampleBuffer) {
            if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                currentBuffer = pixelBuffer
                if !frameSizeCaptured {
                    let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                    let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                    longSide = max(frameWidth, frameHeight)
                    shortSide = min(frameWidth, frameHeight)
                    frameSizeCaptured = true
                }
                
                // Determine image orientation based on device orientation
                let imageOrientation: CGImagePropertyOrientation
                switch UIDevice.current.orientation {
                case .portrait:
                    imageOrientation = .up
                case .portraitUpsideDown:
                    imageOrientation = .down
                case .landscapeLeft:
                    imageOrientation = .up
                case .landscapeRight:
                    imageOrientation = .up
                case .unknown:
                    imageOrientation = .up
                default:
                    imageOrientation = .up
                }
                
                // Create VNImageRequestHandler and perform vision request
                let handler = VNImageRequestHandler(
                    cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
                if UIDevice.current.orientation != .faceUp {  // stop if placed down on a table
                    t0 = CACurrentMediaTime()  // inference start
                    do {
                        try handler.perform([visionRequest])
                    } catch {
                        print(error)
                    }
                    t1 = CACurrentMediaTime() - t0  // inference dt
                }
                
                currentBuffer = nil
            }
        }

        // MARK: - Setup Functions
        func setUpBoundingBoxViews() {
            // Ensure all bounding box views are initialized up to the maximum allowed.
            while boundingBoxViews.count < maxBoundingBoxViews {
                boundingBoxViews.append(BoundingBoxView())
            }
            
            // Retrieve class labels directly from the CoreML model's class labels, if available.
            guard let classLabels = mlModel.modelDescription.classLabels as? [String] else {
                fatalError("Class labels are missing from the model description")
            }
            
            // Assign colors to the classes.
            var count = 0
            for label in classLabels {
                let color = ultralyticsColorsolors[count]
                count += 1
                if count > 19 {
                    count = 0
                }
                colors[label] = color
            }
        }
        
        func startVideo() {
            videoCapture = VideoCapture()
            videoCapture.delegate = self
            
            videoCapture.setUp(sessionPreset: .photo) { success in
                if success {
                    // Add the video preview into the UI.
                    if let previewLayer = self.videoCapture.previewLayer {
                        self.videoPreview.layer.addSublayer(previewLayer)
                        self.videoCapture.previewLayer?.frame = self.videoPreview.bounds
                    }
                    
                    // Add the bounding box layers to the UI, on top of the video preview.
                    for box in self.boundingBoxViews {
                        box.addToLayer(self.videoPreview.layer)
                    }
                    
                    // Once everything is set up, we can start capturing live video.
                    self.videoCapture.start()
                }
            }
        }
        
        func setModel() {
            /// VNCoreMLModel
            detector = try! VNCoreMLModel(for: mlModel)
            detector.featureProvider = ThresholdProvider()
            
            /// VNCoreMLRequest
            let request = VNCoreMLRequest(
                model: detector,
                completionHandler: { [weak self] request, error in
                    self?.processObservations(for: request, error: error)
                })
            request.imageCropAndScaleOption = .scaleFill
            visionRequest = request
            t2 = 0.0  // inference dt smoothed
            t3 = CACurrentMediaTime()  // FPS start
            t4 = 0.0  // FPS dt smoothed
        }
        
        func setLabels() {
            self.labelName.text = "YOLO11m"
            self.labelVersion.text = "Version " + UserDefaults.standard.string(forKey: "app_version")!
        }
        
        private func setUpOrientationChangeNotification() {
            NotificationCenter.default.addObserver(
                self, selector: #selector(orientationDidChange),
                name: UIDevice.orientationDidChangeNotification, object: nil)
        }
        
        @objc func orientationDidChange() {
            videoCapture.updateVideoOrientation()
        }
        
        override func viewWillTransition(
            to size: CGSize, with coordinator: any UIViewControllerTransitionCoordinator
        ) {
            super.viewWillTransition(to: size, with: coordinator)
            
            if size.width > size.height {
                labelSliderConf.isHidden = true
                sliderConf.isHidden = true
                labelSliderIoU.isHidden = true
                sliderIoU.isHidden = true
                toolBar.setBackgroundImage(UIImage(), forToolbarPosition: .any, barMetrics: .default)
                toolBar.setShadowImage(UIImage(), forToolbarPosition: .any)
                
                labelSliderConfLandScape.isHidden = false
                sliderConfLandScape.isHidden = false
                labelSliderIoULandScape.isHidden = false
                sliderIoULandScape.isHidden = false
                
            } else {
                labelSliderConf.isHidden = false
                sliderConf.isHidden = false
                labelSliderIoU.isHidden = false
                sliderIoU.isHidden = false
                toolBar.setBackgroundImage(nil, forToolbarPosition: .any, barMetrics: .default)
                toolBar.setShadowImage(nil, forToolbarPosition: .any)
                
                labelSliderConfLandScape.isHidden = true
                sliderConfLandScape.isHidden = true
                labelSliderIoULandScape.isHidden = true
                sliderIoULandScape.isHidden = true
            }
            self.videoCapture.previewLayer?.frame = CGRect(
                x: 0, y: 0, width: size.width, height: size.height)
        }

        // MARK: - IBActions
        @IBAction func vibrate(_ sender: Any) {
            selection.selectionChanged()
        }
        
        @IBAction func indexChanged(_ sender: Any) {
            selection.selectionChanged()
            activityIndicator.startAnimating()
            
            /// Switch model
            switch segmentedControl.selectedSegmentIndex {
            case 0:
                self.labelName.text = "YOLO11n"
                mlModel = try! yolo11n(configuration: .init()).model
            case 1:
                self.labelName.text = "YOLO11s"
                mlModel = try! yolo11s(configuration: .init()).model
            case 2:
                self.labelName.text = "YOLO11m"
                mlModel = try! yolo11m(configuration: .init()).model
            case 3:
                self.labelName.text = "YOLO11l"
                mlModel = try! yolo11l(configuration: .init()).model
            case 4:
                self.labelName.text = "YOLO11Custom"
                mlModel = try! FENS_model(configuration: .init()).model
            default:
                break
            }
            setModel()
            setUpBoundingBoxViews()
            activityIndicator.stopAnimating()
        }
        
        /// Update thresholds from slider values
        @IBAction func sliderChanged(_ sender: Any) {
            let conf = Double(round(100 * sliderConf.value)) / 100
            let iou = Double(round(100 * sliderIoU.value)) / 100
            self.labelSliderConf.text = String(conf) + " Confidence Threshold"
            self.labelSliderIoU.text = String(iou) + " IoU Threshold"
            detector.featureProvider = ThresholdProvider(iouThreshold: iou, confidenceThreshold: conf)
        }
        
        @IBAction func takePhoto(_ sender: Any?) {
            let t0 = DispatchTime.now().uptimeNanoseconds
            let settings = AVCapturePhotoSettings()
            usleep(20_000)  // short 20 ms delay to allow camera to focus
            self.videoCapture.cameraOutput.capturePhoto(
                with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
            print("Photo taken: ", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1E9)
        }
        
        @IBAction func logoButton(_ sender: Any) {
            selection.selectionChanged()
            if let link = URL(string: "https://www.ultralytics.com") {
                UIApplication.shared.open(link)
            }
        }
        
        @IBAction func playButton(_ sender: Any) {
            selection.selectionChanged()
            self.videoCapture.start()
            playButtonOutlet.isEnabled = false
            pauseButtonOutlet.isEnabled = true
        }
        
        @IBAction func pauseButton(_ sender: Any?) {
            selection.selectionChanged()
            self.videoCapture.stop()
            playButtonOutlet.isEnabled = true
            pauseButtonOutlet.isEnabled = false
        }
        
        @IBAction func switchCameraTapped(_ sender: Any) {
            self.videoCapture.captureSession.beginConfiguration()
            let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput
            self.videoCapture.captureSession.removeInput(currentInput!)
            guard let currentPosition = currentInput?.device.position else { return }
            
            let nextCameraPosition: AVCaptureDevice.Position = currentPosition == .back ? .front : .back
            
            let newCameraDevice = bestCaptureDevice(for: nextCameraPosition)
            
            guard let videoInput1 = try? AVCaptureDeviceInput(device: newCameraDevice) else {
                return
            }
            
            self.videoCapture.captureSession.addInput(videoInput1)
            self.videoCapture.updateVideoOrientation()
            
            self.videoCapture.captureSession.commitConfiguration()
        }
        
        @IBAction func shareButton(_ sender: Any) {
            selection.selectionChanged()
            let settings = AVCapturePhotoSettings()
            self.videoCapture.cameraOutput.capturePhoto(
                with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
        }
        
        @IBAction func saveScreenshotButton(_ shouldSave: Bool = true) {
            // Screenshot functionality can be implemented here if needed
        }

        // MARK: - Pinch to Zoom
        @IBAction func pinch(_ pinch: UIPinchGestureRecognizer) {
            let device = videoCapture.captureDevice

            // Return zoom value between the minimum and maximum zoom values
            func minMaxZoom(_ factor: CGFloat) -> CGFloat {
              return min(min(max(factor, minimumZoom), maximumZoom), device.activeFormat.videoMaxZoomFactor)
            }

            func update(scale factor: CGFloat) {
              do {
                try device.lockForConfiguration()
                defer {
                  device.unlockForConfiguration()
                }
                device.videoZoomFactor = factor
              } catch {
                print("\(error.localizedDescription)")
              }
            }

            let newScaleFactor = minMaxZoom(pinch.scale * lastZoomFactor)
            switch pinch.state {
            case .began, .changed:
              update(scale: newScaleFactor)
              self.labelZoom.text = String(format: "%.2fx", newScaleFactor)
              self.labelZoom.font = UIFont.preferredFont(forTextStyle: .title2)
            case .ended:
              lastZoomFactor = minMaxZoom(newScaleFactor)
              update(scale: lastZoomFactor)
              self.labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
            default: break
            }
        }
        
        // MARK: - Utility Functions
        func saveText(text: String, file: String = "saved.txt") {
            if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
                let fileURL = dir.appendingPathComponent(file)
                
                // Writing
                do {  // Append to file if it exists
                    let fileHandle = try FileHandle(forWritingTo: fileURL)
                    fileHandle.seekToEndOfFile()
                    fileHandle.write(text.data(using: .utf8)!)
                    fileHandle.closeFile()
                } catch {  // Create new file and write
                    do {
                        try text.write(to: fileURL, atomically: false, encoding: .utf8)
                    } catch {
                        print("no file written")
                    }
                }
            }
        }
        
        func saveImage() {
            let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            let fileURL = dir!.appendingPathComponent("saved.jpg")
            let image = UIImage(named: "ultralytics_yolo_logotype.png")
            FileManager.default.createFile(
                atPath: fileURL.path, contents: image!.jpegData(compressionQuality: 0.5), attributes: nil)
        }
        
        func freeSpace() -> Double {
            let fileURL = URL(fileURLWithPath: NSHomeDirectory() as String)
            do {
                let values = try fileURL.resourceValues(forKeys: [
                    .volumeAvailableCapacityForImportantUsageKey
                ])
                return Double(values.volumeAvailableCapacityForImportantUsage!) / 1E9  // Bytes to GB
            } catch {
                print("Error retrieving storage capacity: \(error.localizedDescription)")
            }
            return 0
        }
        
        func memoryUsage() -> Double {
            var taskInfo = mach_task_basic_info()
            var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
            let kerr: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
                $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                    task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
                }
            }
            if kerr == KERN_SUCCESS {
                return Double(taskInfo.resident_size) / 1E9  // Bytes to GB
            } else {
                return 0
            }
        }
        
        func bestCaptureDevice(for position: AVCaptureDevice.Position) -> AVCaptureDevice {
            if #available(iOS 15.0, *) {
                let discoverySession = AVCaptureDevice.DiscoverySession(
                    deviceTypes: [.builtInTripleCamera, .builtInDualCamera, .builtInDualWideCamera, .builtInWideAngleCamera],
                    mediaType: .video,
                    position: position
                )
                return discoverySession.devices.first ?? AVCaptureDevice.default(for: .video)!
            } else {
                return AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) ?? AVCaptureDevice.default(for: .video)!
            }
        }
    }

    // MARK: - VideoCaptureDelegate Extension
    extension ViewController: VideoCaptureDelegate {
        func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
            predict(sampleBuffer: sampleBuffer)
        }
    }

    // MARK: - AVSpeechSynthesizerDelegate Extension
    extension ViewController: AVSpeechSynthesizerDelegate {
        func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
            isSpeaking = false
        }
        
        func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
            isSpeaking = false
        }
    }

    // MARK: - AVCapturePhotoCaptureDelegate Extension
    extension ViewController: AVCapturePhotoCaptureDelegate {
        func photoOutput(
            _ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
        ) {
            if let error = error {
                print("error occurred : \(error.localizedDescription)")
            }
            if let dataImage = photo.fileDataRepresentation() {
                let dataProvider = CGDataProvider(data: dataImage as CFData)
                let cgImageRef: CGImage! = CGImage(
                    jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true,
                    intent: .defaultIntent)
                var isCameraFront = false
                if let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput,
                   currentInput.device.position == .front
                {
                    isCameraFront = true
                }
                var orientation: CGImagePropertyOrientation = isCameraFront ? .leftMirrored : .right
                switch UIDevice.current.orientation {
                case .landscapeLeft:
                    orientation = isCameraFront ? .downMirrored : .up
                case .landscapeRight:
                    orientation = isCameraFront ? .upMirrored : .down
                default:
                    break
                }
                var image = UIImage(cgImage: cgImageRef, scale: 0.5, orientation: .right)
                if let orientedCIImage = CIImage(image: image)?.oriented(orientation),
                   let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent)
                {
                    image = UIImage(cgImage: cgImage)
                }
                let imageView = UIImageView(image: image)
                imageView.contentMode = .scaleAspectFill
                imageView.frame = videoPreview.frame
                let imageLayer = imageView.layer
                videoPreview.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)
                
                let bounds = UIScreen.main.bounds
                UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
                self.View0.drawHierarchy(in: bounds, afterScreenUpdates: true)
                let img = UIGraphicsGetImageFromCurrentImageContext()
                UIGraphicsEndImageContext()
                imageLayer.removeFromSuperlayer()
                let activityViewController = UIActivityViewController(
                    activityItems: [img!], applicationActivities: nil)
                activityViewController.popoverPresentationController?.sourceView = self.View0
                self.present(activityViewController, animated: true, completion: nil)
            } else {
                print("AVCapturePhotoCaptureDelegate Error")
            }
        }
    }
