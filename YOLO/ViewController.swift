//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  Main View Controller for Ultralytics YOLO App
//  This file is part of the Ultralytics YOLO app, enabling real-time object detection using YOLO11 models on iOS devices.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  This ViewController manages the app's main screen, handling video capture, model selection, detection visualization,
//  and user interactions. It sets up and controls the video preview layer, handles model switching via a segmented control,
//  manages UI elements like sliders for confidence and IoU thresholds, and displays detection results on the video feed.
//  It leverages CoreML, Vision, and AVFoundation frameworks to perform real-time object detection and to interface with
//  the device's camera.

import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision
import Speech

var mlModel = try! yolo11m(configuration: mlmodelConfig).model
var mlmodelConfig: MLModelConfiguration = {
  let config = MLModelConfiguration()

  if #available(iOS 17.0, *) {
    config.setValue(1, forKey: "experimentalMLE5EngineUsage")
  }

  return config
}()

class ViewController: UIViewController {
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
  // var cameraOutput: AVCapturePhotoOutput!
  var longSide: CGFloat = 3
  var shortSide: CGFloat = 4
  var frameSizeCaptured = false
  // Sesli bildirim için
  let speechSynthesizer = AVSpeechSynthesizer()
  var platesQueue: [(rect: CGRect, label: String)] = []
  var isReading = false
  var lastProximityWarningTime: TimeInterval = 0
  let proximityWarningCooldown: TimeInterval = 5.0 // Seconds between warnings
  let proximityThreshold: CGFloat = 0.70 // License plate must cover at least 70% of screen width or height
    
  private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))!
  private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
  private var recognitionTask: SFSpeechRecognitionTask?
  private let audioEngine = AVAudioEngine()
  private var isVoiceCommandMenuActive = false
  private var voiceCommandMicIndicator: UIView?



  // Developer mode
  let developerMode = UserDefaults.standard.bool(forKey: "developer_mode")  // developer mode selected in settings
  let save_detections = false  // write every detection to detections.txt
  let save_frames = false  // write every frame to frames.txt

  lazy var visionRequest: VNCoreMLRequest = {
    let request = VNCoreMLRequest(
      model: detector,
      completionHandler: {
        [weak self] request, error in
        self?.processObservations(for: request, error: error)
      })
    // NOTE: BoundingBoxView object scaling depends on request.imageCropAndScaleOption https://developer.apple.com/documentation/vision/vnimagecropandscaleoption
    request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
    return request
  }()
    
    
    // MARK: - Voice Command Actions

    
    // Similar improvements for other plate-finding functions
    func findNearestLicensePlate() {
        var nearestPlate: (rect: CGRect, distance: CGFloat, position: String)? = nil
        let centerX = videoPreview.bounds.width / 2
        let centerY = videoPreview.bounds.height / 2
        let centerPoint = CGPoint(x: centerX, y: centerY)
        
        // Check all visible bounding boxes
        for i in 0..<boundingBoxViews.count {
            if !boundingBoxViews[i].shapeLayer.isHidden {
                // Get label text to check if it's a license plate
                let boxLabel = boundingBoxViews[i].textLayer.string as? String ?? ""
                
                if boxLabel.lowercased().contains("license_plate") {
                    // Get frame from the shape layer's path
                    if let path = boundingBoxViews[i].shapeLayer.path {
                        let boxFrame = UIBezierPath(cgPath: path).bounds
                        
                        // Calculate distance from center of screen to center of box
                        let boxCenterX = boxFrame.midX
                        let boxCenterY = boxFrame.midY
                        let boxCenter = CGPoint(x: boxCenterX, y: boxCenterY)
                        
                        let dx = centerPoint.x - boxCenter.x
                        let dy = centerPoint.y - boxCenter.y
                        let distance = sqrt(dx*dx + dy*dy)
                        
                        // Get position description
                        let position = determinePlatePosition(rect: boxFrame, viewWidth: videoPreview.bounds.width)
                        
                        // If this is the first or closer than current nearest
                        if nearestPlate == nil || distance < nearestPlate!.distance {
                            nearestPlate = (boxFrame, distance, position)
                        }
                    }
                }
            }
        }
        
        if let plate = nearestPlate {
            // More detailed directional guidance based on position
            let directionGuidance = getDirectionalGuidance(for: plate.position)
            speakText("Nearest license plate is \(plate.position). \(directionGuidance)")
        } else {
            speakText("No license plates detected")
        }
        
        // Continue voice recognition after a short delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            if self.isVoiceCommandMenuActive {
                self.startSpeechRecognition()
            }
        }
    }

    // Similar improvements for other plate-finding functions
    func findFarthestLicensePlate() {
        var farthestPlate: (rect: CGRect, distance: CGFloat, position: String)? = nil
        let centerX = videoPreview.bounds.width / 2
        let centerY = videoPreview.bounds.height / 2
        let centerPoint = CGPoint(x: centerX, y: centerY)
        
        // Check all visible bounding boxes
        for i in 0..<boundingBoxViews.count {
            if !boundingBoxViews[i].shapeLayer.isHidden {
                // Get label text to check if it's a license plate
                let boxLabel = boundingBoxViews[i].textLayer.string as? String ?? ""
                
                if boxLabel.lowercased().contains("license_plate") {
                    // Get frame from the shape layer's path
                    if let path = boundingBoxViews[i].shapeLayer.path {
                        let boxFrame = UIBezierPath(cgPath: path).bounds
                        
                        // Calculate distance from center of screen to center of box
                        let boxCenterX = boxFrame.midX
                        let boxCenterY = boxFrame.midY
                        let boxCenter = CGPoint(x: boxCenterX, y: boxCenterY)
                        
                        let dx = centerPoint.x - boxCenter.x
                        let dy = centerPoint.y - boxCenter.y
                        let distance = sqrt(dx*dx + dy*dy)
                        
                        // Get position description
                        let position = determinePlatePosition(rect: boxFrame, viewWidth: videoPreview.bounds.width)
                        
                        // If this is the first or farther than current farthest
                        if farthestPlate == nil || distance > farthestPlate!.distance {
                            farthestPlate = (boxFrame, distance, position)
                        }
                    }
                }
            }
        }
        
        if let plate = farthestPlate {
            // More detailed directional guidance based on position
            let directionGuidance = getDirectionalGuidance(for: plate.position)
            speakText("Farthest license plate is \(plate.position). \(directionGuidance)")
        } else {
            speakText("No license plates detected")
        }
        
        // Continue voice recognition after a short delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            if self.isVoiceCommandMenuActive {
                self.startSpeechRecognition()
            }
        }
    }

    
    func findRandomLicensePlate() {
        var licensePlates: [(rect: CGRect, position: String)] = []
        
        // Collect all license plates
        for i in 0..<boundingBoxViews.count {
            if !boundingBoxViews[i].shapeLayer.isHidden {
                // Get label text to check if it's a license plate
                let boxLabel = boundingBoxViews[i].textLayer.string as? String ?? ""
                
                if boxLabel.lowercased().contains("license_plate") {
                    // Get frame from the shape layer's path
                    if let path = boundingBoxViews[i].shapeLayer.path {
                        let boxFrame = UIBezierPath(cgPath: path).bounds
                        let position = determinePlatePosition(rect: boxFrame, viewWidth: videoPreview.bounds.width)
                        licensePlates.append((boxFrame, position))
                    }
                }
            }
        }
        
        if !licensePlates.isEmpty {
            // Pick a random license plate
            let randomIndex = Int.random(in: 0..<licensePlates.count)
            let randomPlate = licensePlates[randomIndex]
            let directionGuidance = getDirectionalGuidance(for: randomPlate.position)
            speakText("Random license plate is \(randomPlate.position). \(directionGuidance)")
        } else {
            speakText("No license plates detected")
        }
        
        // Continue voice recognition after a short delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            if self.isVoiceCommandMenuActive {
                self.startSpeechRecognition()
            }
        }
    }

    func countLicensePlates() {
        var count = 0
        
        // Count all license plates
        for i in 0..<boundingBoxViews.count {
            if !boundingBoxViews[i].shapeLayer.isHidden {
                // Get label text to check if it's a license plate
                let boxLabel = boundingBoxViews[i].textLayer.string as? String ?? ""
                
                if boxLabel.lowercased().contains("license_plate") {
                    count += 1
                }
            }
        }
        
        if count == 0 {
            speakText("No license plates detected")
        } else if count == 1 {
            speakText("There is 1 license plate detected")
        } else {
            speakText("There are \(count) license plates detected")
        }
        
        // Continue voice recognition
        startSpeechRecognition()
    }

    // Improved help information
    func provideHelpInformation() {
        let helpText = """
        Available commands:
        Take me to the nearest license plate - I'll guide you to the closest license plate
        Take me to the farthest license plate - I'll guide you to the most distant license plate
        Take me to a random license plate - I'll guide you to a randomly selected license plate
        How many license plates are detected - I'll count visible license plates
        Help - Repeats this information
        Exit voice command menu - Closes this menu and returns to normal operation
        """
        
        speakText(helpText)
        
        print("Help information provided, will restart speech recognition after speech completes")
        
        // Continue voice recognition after speech finishes
        // This special handling for help command prevents the recognition loop problem
        if !speechSynthesizer.isSpeaking {
            DispatchQueue.main.asyncAfter(deadline: .now() + 10.0) {
                if self.isVoiceCommandMenuActive {
                    print("Starting new recognition session after help")
                    self.startSpeechRecognition()
                }
            }
        }
    }
    
    
    func processVoiceCommand(_ command: String) {
        // Convert command to lowercase for easier matching
        let lowerCommand = command.lowercased()
        print("Processing command: \(lowerCommand)")
        
        // Find nearest license plate
        if lowerCommand.contains("nearest") && lowerCommand.contains("license plate") ||
           lowerCommand.contains("en yakın") && lowerCommand.contains("plaka") {
            findNearestLicensePlate()
        }
        // Find farthest license plate
        else if lowerCommand.contains("farthest") && lowerCommand.contains("license plate") ||
                lowerCommand.contains("en uzak") && lowerCommand.contains("plaka") {
            findFarthestLicensePlate()
        }
        // Find random license plate
        else if lowerCommand.contains("random") && lowerCommand.contains("license plate") ||
                lowerCommand.contains("rastgele") && lowerCommand.contains("plaka") {
            findRandomLicensePlate()
        }
        // Count license plates
        else if lowerCommand.contains("how many") && lowerCommand.contains("license plate") ||
                lowerCommand.contains("kaç") && lowerCommand.contains("plaka") {
            countLicensePlates()
        }
        // Help command
        else if lowerCommand.contains("help") || lowerCommand.contains("yardım") ||
                lowerCommand.contains("what can i") || lowerCommand.contains("ne yapabilirim") {
            provideHelpInformation()
        }
        // Exit voice command menu
        else if lowerCommand.contains("exit") || lowerCommand.contains("quit") ||
                lowerCommand.contains("çık") || lowerCommand.contains("kapat") {
            speakText("Exiting voice command menu")
            stopVoiceCommandMenu()
            return
        }
        // Command not recognized
        else {
            speakText("Command not recognized. Say help for available commands.")
        }
        
        // For commands other than "exit", restart speech recognition after a short delay
        if isVoiceCommandMenuActive && !lowerCommand.contains("exit") && !lowerCommand.contains("quit") &&
           !lowerCommand.contains("çık") && !lowerCommand.contains("kapat") {
            DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
                if self.isVoiceCommandMenuActive {
                    print("Restarting speech recognition after command")
                    self.startSpeechRecognition()
                }
            }
        }
    }

    // Updated voice command menu activation
    func startVoiceCommandMenu() {
        print("Attempting to start voice command menu with fixed configuration")
        
        // First check to see if we're already in voice command mode
        if isVoiceCommandMenuActive {
            speakText("Voice command menu is already active")
            return
        }
        
        // Reset audio engine and recognition task before starting
        audioEngine.stop()
        if audioEngine.inputNode.numberOfInputs > 0 {
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        
        if recognitionTask != nil {
            recognitionTask?.cancel()
            recognitionTask = nil
        }
        
        if recognitionRequest != nil {
            recognitionRequest?.endAudio()
            recognitionRequest = nil
        }
        
        // Request speech recognition authorization with better error handling
        SFSpeechRecognizer.requestAuthorization { [weak self] authStatus in
            guard let self = self else { return }
            
            DispatchQueue.main.async {
                switch authStatus {
                case .authorized:
                    print("Speech recognition authorized, activating voice command menu")
                    self.isVoiceCommandMenuActive = true
                    self.showMicrophoneIndicator()
                    self.speakText("Welcome to voice command menu. Say help to hear available commands.")
                    
                    // Start speech recognition after a short delay to allow welcome message to be spoken
                    DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                        print("Starting initial speech recognition")
                        self.startSpeechRecognition()
                    }
                    
                case .denied:
                    print("Speech recognition permission denied")
                    self.speakText("Speech recognition permission denied. Please enable microphone access in Settings.")
                    
                case .restricted, .notDetermined:
                    print("Speech recognition not available or not determined")
                    self.speakText("Speech recognition not available. Please try again.")
                    
                @unknown default:
                    print("Unknown speech recognition authorization status")
                    self.speakText("Speech recognition not available. Please try again.")
                }
            }
        }
    }

    func showMicrophoneIndicator() {
        // Create a visual indicator for active voice command mode
        if voiceCommandMicIndicator == nil {
            let indicator = UIView(frame: CGRect(x: 0, y: 0, width: 60, height: 60))
            indicator.backgroundColor = UIColor.red.withAlphaComponent(0.5)
            indicator.layer.cornerRadius = 30
            indicator.center = CGPoint(x: videoPreview.bounds.width - 50, y: 70)
            
            // Add microphone icon or label if needed
            let micLabel = UILabel(frame: indicator.bounds)
            micLabel.text = "🎤"
            micLabel.textAlignment = .center
            micLabel.font = UIFont.systemFont(ofSize: 30)
            indicator.addSubview(micLabel)
            
            // Add pulsating animation
            let pulseAnimation = CABasicAnimation(keyPath: "transform.scale")
            pulseAnimation.duration = 1.0
            pulseAnimation.fromValue = 0.9
            pulseAnimation.toValue = 1.1
            pulseAnimation.autoreverses = true
            pulseAnimation.repeatCount = Float.infinity
            indicator.layer.add(pulseAnimation, forKey: "pulse")
            
            videoPreview.addSubview(indicator)
            voiceCommandMicIndicator = indicator
        }
    }

    func hideMicrophoneIndicator() {
        voiceCommandMicIndicator?.removeFromSuperview()
        voiceCommandMicIndicator = nil
    }

    // Improved audio session configuration
    // Fixed startSpeechRecognition function with proper audio session configuration
    func startSpeechRecognition() {
        print("Starting speech recognition with fixed audio session")
        
        // Check if there's an existing task running and cancel it
        if recognitionTask != nil {
            recognitionTask?.cancel()
            recognitionTask = nil
        }
        
        // Create an audio session with fixed configuration
        let audioSession = AVAudioSession.sharedInstance()
        do {
            // Use playAndRecord category which is compatible with defaultToSpeaker option
            try audioSession.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Audio session setup failed: \(error)")
            speakText("Could not start voice recognition. Please try again.")
            return
        }
        
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        
        // Check if device has a recognition engine
        guard let recognitionRequest = recognitionRequest else {
            speakText("Speech recognition not available on this device")
            return
        }
        
        // We want to get results while the user is speaking
        recognitionRequest.shouldReportPartialResults = true
        
        // Start recognition with better error handling
        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            
            var isFinal = false
            
            if let result = result {
                // Get the transcript
                let recognizedText = result.bestTranscription.formattedString
                print("Recognized: \(recognizedText)")
                isFinal = result.isFinal
                
                // If we have a complete command, process it
                if isFinal && recognizedText.count > 0 {
                    print("Final recognition result: \(recognizedText)")
                    
                    // Immediately stop audio engine
                    self.audioEngine.stop()
                    self.audioEngine.inputNode.removeTap(onBus: 0)
                    self.recognitionRequest = nil
                    
                    // Process the command
                    DispatchQueue.main.async {
                        self.processVoiceCommand(recognizedText)
                    }
                }
            }
            
            if error != nil {
                print("Recognition error: \(error!)")
                // Only restart if it's a non-fatal error
                if self.isVoiceCommandMenuActive {
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                        self.startSpeechRecognition()
                    }
                }
            } else if isFinal {
                // Already handled above
            }
        }
        
        // Configure the audio input with error handling
        let recordingFormat = audioEngine.inputNode.outputFormat(forBus: 0)
        
        // Make sure we remove any existing tap first
        if audioEngine.inputNode.numberOfInputs > 0 {
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        
        audioEngine.inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            self.recognitionRequest?.append(buffer)
        }
        
        // Start the audio engine with better error handling
        do {
            audioEngine.prepare()
            try audioEngine.start()
            print("Audio engine started successfully")
        } catch {
            print("Audio engine failed to start: \(error)")
            speakText("Could not start voice recognition. Please try again.")
        }
    }
    
    // Fixed stop voice command menu function
    func stopVoiceCommandMenu() {
        print("Stopping voice command menu with proper cleanup")
        isVoiceCommandMenuActive = false
        hideMicrophoneIndicator()
        
        // Stop speech recognition and clean up
        if recognitionTask != nil {
            recognitionTask?.cancel()
            recognitionTask = nil
        }
        
        if recognitionRequest != nil {
            recognitionRequest?.endAudio()
            recognitionRequest = nil
        }
        
        audioEngine.stop()
        if audioEngine.inputNode.numberOfInputs > 0 {
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        
        // Reset audio session with better error handling
        do {
            // Set to ambient which is least likely to cause issues
            try AVAudioSession.sharedInstance().setCategory(.ambient)
            try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        } catch {
            print("Failed to deactivate audio session: \(error)")
            // No need for further error handling here
        }
        
        speakText("Voice command menu closed")
    }


    @objc func handleTripleTap() {
        print("Triple tap detected")
        
        // Eğer zaten sesli komut menüsü aktifse, kapat
        if isVoiceCommandMenuActive {
            stopVoiceCommandMenu()
            return
        }
        
        // Sesli komut menüsünü başlat
        startVoiceCommandMenu()
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        slider.value = 30
        setLabels()
        setUpBoundingBoxViews()
        setUpOrientationChangeNotification()
        
        // Request speech recognition authorization
        SFSpeechRecognizer.requestAuthorization { authStatus in
          // Do nothing here, we'll handle authorization in startVoiceCommandMenu
        }
        
        // Double tap gesture recognizer ekleme
        let doubleTapGesture = UITapGestureRecognizer(target: self, action: #selector(handleDoubleTap))
        doubleTapGesture.numberOfTapsRequired = 2
        doubleTapGesture.delaysTouchesBegan = true
        
        // Triple tap gesture recognizer ekleme
        let tripleTapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTripleTap))
        tripleTapGesture.numberOfTapsRequired = 3
        tripleTapGesture.delaysTouchesBegan = true
        
        // Triple tap'in double tap ile çakışmasını önle
        doubleTapGesture.require(toFail: tripleTapGesture)
          
        // Varsa diğer gesture recognizer'lar ile çakışmayı önle
        if let existingGestures = videoPreview.gestureRecognizers {
            for gesture in existingGestures {
                if let tapGesture = gesture as? UITapGestureRecognizer {
                    tapGesture.require(toFail: doubleTapGesture)
                    tapGesture.require(toFail: tripleTapGesture)
                }
            }
        }
          
        videoPreview.addGestureRecognizer(doubleTapGesture)
        videoPreview.addGestureRecognizer(tripleTapGesture)
        videoPreview.isUserInteractionEnabled = true
        videoPreview.isMultipleTouchEnabled = true
        
        startVideo()
        // setModel()
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

  private func setUpOrientationChangeNotification() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(orientationDidChange),
      name: UIDevice.orientationDidChangeNotification, object: nil)
  }

  @objc func orientationDidChange() {
    videoCapture.updateVideoOrientation()
    //      frameSizeCaptured = false
  }

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
      self.labelName.text = "YOLO11x"
      mlModel = try! yolo11x(configuration: .init()).model
    case 5:
      self.labelName.text = "YOLObest"
      mlModel = try! BestLicensePlateModel(configuration: .init()).model
    default:
      break
    }
    setModel()
    setUpBoundingBoxViews()
    activityIndicator.stopAnimating()
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
    request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
    visionRequest = request
    t2 = 0.0  // inference dt smoothed
    t3 = CACurrentMediaTime()  // FPS start
    t4 = 0.0  // FPS dt smoothed
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

    // 1. captureSession and cameraOutput
    // session = videoCapture.captureSession  // session = AVCaptureSession()
    // session.sessionPreset = AVCaptureSession.Preset.photo
    // cameraOutput = AVCapturePhotoOutput()
    // cameraOutput.isHighResolutionCaptureEnabled = true
    // cameraOutput.isDualCameraDualPhotoDeliveryEnabled = true
    // print("1 Done: ", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1E9)

    // 2. Settings
    let settings = AVCapturePhotoSettings()
    // settings.flashMode = .off
    // settings.isHighResolutionPhotoEnabled = cameraOutput.isHighResolutionCaptureEnabled
    // settings.isDualCameraDualPhotoDeliveryEnabled = self.videoCapture.cameraOutput.isDualCameraDualPhotoDeliveryEnabled

    // 3. Capture Photo
    usleep(20_000)  // short 10 ms delay to allow camera to focus
    self.videoCapture.cameraOutput.capturePhoto(
      with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
    print("3 Done: ", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1E9)
  }

  @IBAction func logoButton(_ sender: Any) {
    selection.selectionChanged()
    if let link = URL(string: "https://www.ultralytics.com") {
      UIApplication.shared.open(link)
    }
  }

  func setLabels() {
    self.labelName.text = "YOLO11m"
    self.labelVersion.text = "Version " + UserDefaults.standard.string(forKey: "app_version")!
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

  // share image
  @IBAction func shareButton(_ sender: Any) {
    selection.selectionChanged()
    let settings = AVCapturePhotoSettings()
    self.videoCapture.cameraOutput.capturePhoto(
      with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
  }

  // share screenshot
  @IBAction func saveScreenshotButton(_ shouldSave: Bool = true) {
    // let layer = UIApplication.shared.keyWindow!.layer
    // let scale = UIScreen.main.scale
    // UIGraphicsBeginImageContextWithOptions(layer.frame.size, false, scale);
    // layer.render(in: UIGraphicsGetCurrentContext()!)
    // let screenshot = UIGraphicsGetImageFromCurrentImageContext()
    // UIGraphicsEndImageContext()

    // let screenshot = UIApplication.shared.screenShot
    // UIImageWriteToSavedPhotosAlbum(screenshot!, nil, nil, nil)
  }

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

  func setUpBoundingBoxViews() {
    // Ensure all bounding box views are initialized up to the maximum allowed.
    while boundingBoxViews.count < maxBoundingBoxViews {
      boundingBoxViews.append(BoundingBoxView())
    }

    // Retrieve class labels directly from the CoreML model's class labels, if available.
    guard let classLabels = mlModel.modelDescription.classLabels as? [String] else {
      fatalError("Class labels are missing from the model description")
    }

    // Assign random colors to the classes.
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
      // .hd4K3840x2160 or .photo (4032x3024)  Warning: 4k may not work on all devices i.e. 2019 iPod
      if success {
        // Add the video preview into the UI.
        if let previewLayer = self.videoCapture.previewLayer {
          self.videoPreview.layer.addSublayer(previewLayer)
          self.videoCapture.previewLayer?.frame = self.videoPreview.bounds  // resize preview layer
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
      /// - Tag: MappingOrientation
      // The frame is always oriented based on the camera sensor,
      // so in most cases Vision needs to rotate it for the model to work as expected.
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

      // Invoke a VNRequestHandler with that image
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

  func processObservations(for request: VNRequest, error: Error?) {
    DispatchQueue.main.async {
      if let results = request.results as? [VNRecognizedObjectObservation] {
        self.show(predictions: results)
      } else {
        self.show(predictions: [])
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

  // Save text file
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

      // Reading
      // do {let text2 = try String(contentsOf: fileURL, encoding: .utf8)} catch {/* error handling here */}
    }
  }

  // Save image file
  func saveImage() {
    let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    let fileURL = dir!.appendingPathComponent("saved.jpg")
    let image = UIImage(named: "ultralytics_yolo_logotype.png")
    FileManager.default.createFile(
      atPath: fileURL.path, contents: image!.jpegData(compressionQuality: 0.5), attributes: nil)
  }

  // Return hard drive space (GB)
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

  // Return RAM usage (GB)
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

    func show(predictions: [VNRecognizedObjectObservation]) {
      var str = ""
      // date
      let date = Date()
      let calendar = Calendar.current
      let hour = calendar.component(.hour, from: date)
      let minutes = calendar.component(.minute, from: date)
      let seconds = calendar.component(.second, from: date)
      let nanoseconds = calendar.component(.nanosecond, from: date)
      let sec_day =
        Double(hour) * 3600.0 + Double(minutes) * 60.0 + Double(seconds) + Double(nanoseconds) / 1E9  // seconds in the day

      self.labelSlider.text =
        String(predictions.count) + " items (max " + String(Int(slider.value)) + ")"
      let width = videoPreview.bounds.width  // 375 pix
      let height = videoPreview.bounds.height  // 812 pix

      if UIDevice.current.orientation == .portrait {

        // ratio = videoPreview AR divided by sessionPreset AR
        var ratio: CGFloat = 1.0
        if videoCapture.captureSession.sessionPreset == .photo {
          ratio = (height / width) / (4.0 / 3.0)  // .photo
        } else {
          ratio = (height / width) / (16.0 / 9.0)  // .hd4K3840x2160, .hd1920x1080, .hd1280x720 etc.
        }

        for i in 0..<boundingBoxViews.count {
          if i < predictions.count && i < Int(slider.value) {
            let prediction = predictions[i]

            var rect = prediction.boundingBox  // normalized xywh, origin lower left
            switch UIDevice.current.orientation {
            case .portraitUpsideDown:
              rect = CGRect(
                x: 1.0 - rect.origin.x - rect.width,
                y: 1.0 - rect.origin.y - rect.height,
                width: rect.width,
                height: rect.height)
            case .landscapeLeft:
              rect = CGRect(
                x: rect.origin.x,
                y: rect.origin.y,
                width: rect.width,
                height: rect.height)
            case .landscapeRight:
              rect = CGRect(
                x: rect.origin.x,
                y: rect.origin.y,
                width: rect.width,
                height: rect.height)
            case .unknown:
              print("The device orientation is unknown, the predictions may be affected")
              fallthrough
            default: break
            }

            if ratio >= 1 {  // iPhone ratio = 1.218
              let offset = (1 - ratio) * (0.5 - rect.minX)
              let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
              rect = rect.applying(transform)
              rect.size.width *= ratio
            } else {  // iPad ratio = 0.75
              let offset = (ratio - 1) * (0.5 - rect.maxY)
              let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
              rect = rect.applying(transform)
              ratio = (height / width) / (3.0 / 4.0)
              rect.size.height /= ratio
            }

            // Scale normalized to pixels [375, 812] [width, height]
            rect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))

            // The labels array is a list of VNClassificationObservation objects,
            // with the highest scoring class first in the list.
            let bestClass = prediction.labels[0].identifier
            let confidence = prediction.labels[0].confidence
            // print(confidence, rect)  // debug (confidence, xywh) with xywh origin top left (pixels)
            let label = String(format: "%@ %.1f", bestClass, confidence * 100)
            let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
            // Show the bounding box.
            boundingBoxViews[i].show(
              frame: rect,
              label: label,
              color: colors[bestClass] ?? UIColor.white,
              alpha: alpha)  // alpha 0 (transparent) to 1 (opaque) for conf threshold 0.2 to 1.0)
            
            // Check for license plate proximity (large bounding box)
            if bestClass.lowercased().contains("license_plate") && confidence > 0.5 {
                // Calculate relative size of the plate compared to the screen
                let boxArea = rect.width * rect.height
                let screenArea = width * height
                let relativeSizeRatio = boxArea / screenArea
                
                // Or check if either dimension is large compared to screen
                let widthRatio = rect.width / width
                let heightRatio = rect.height / height
                
                // Determine if this license plate is too close
                if widthRatio > proximityThreshold || heightRatio > proximityThreshold {
                    // Only warn once every few seconds
                    let currentTime = CACurrentMediaTime()
                    if currentTime - lastProximityWarningTime > proximityWarningCooldown {
                        lastProximityWarningTime = currentTime
                        
                        // Get plate position
                        let position = determinePlatePosition(rect: rect, viewWidth: width)
                        
                        // Alert user about nearby license plate
                        let warningText = "Attention, license plate \(position) very close"
                        speakText(warningText)
                    }
                }
            }

            if developerMode {
              // Write
              if save_detections {
                str += String(
                  format: "%.3f %.3f %.3f %@ %.2f %.1f %.1f %.1f %.1f\n",
                  sec_day, freeSpace(), UIDevice.current.batteryLevel, bestClass, confidence,
                  rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)
              }
            }
          } else {
            boundingBoxViews[i].hide()
          }
        }
      } else {
          let frameAspectRatio = longSide / shortSide
          let viewAspectRatio = width / height
          var scaleX: CGFloat = 1.0
          var scaleY: CGFloat = 1.0
          var offsetX: CGFloat = 0.0
          var offsetY: CGFloat = 0.0
          
          if frameAspectRatio > viewAspectRatio {
              scaleY = height / shortSide
              scaleX = scaleY
              offsetX = (longSide * scaleX - width) / 2
          } else {
              scaleX = width / longSide
              scaleY = scaleX
              offsetY = (shortSide * scaleY - height) / 2
          }
          
          for i in 0..<boundingBoxViews.count {
                  if i < predictions.count {
                    let prediction = predictions[i]

                    var rect = prediction.boundingBox

                    rect.origin.x = rect.origin.x * longSide * scaleX - offsetX
                    rect.origin.y =
                      height
                      - (rect.origin.y * shortSide * scaleY - offsetY + rect.size.height * shortSide * scaleY)
                    rect.size.width *= longSide * scaleX
                    rect.size.height *= shortSide * scaleY

                    let bestClass = prediction.labels[0].identifier
                    let confidence = prediction.labels[0].confidence

                    let label = String(format: "%@ %.1f", bestClass, confidence * 100)
                    let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
                    // Show the bounding box.
                    boundingBoxViews[i].show(
                      frame: rect,
                      label: label,
                      color: colors[bestClass] ?? UIColor.white,
                      alpha: alpha)  // alpha 0 (transparent) to 1 (opaque) for conf threshold 0.2 to 1.0)
                      
                    // Check for license plate proximity (large bounding box)
                    if bestClass.lowercased().contains("license_plate") && confidence > 0.5 {
                        // Calculate relative size of the plate compared to the screen
                        let boxArea = rect.width * rect.height
                        let screenArea = width * height
                        let relativeSizeRatio = boxArea / screenArea
                        
                        // Or check if either dimension is large compared to screen
                        let widthRatio = rect.width / width
                        let heightRatio = rect.height / height
                        
                        // Determine if this license plate is too close
                        if widthRatio > proximityThreshold || heightRatio > proximityThreshold {
                            // Only warn once every few seconds
                            let currentTime = CACurrentMediaTime()
                            if currentTime - lastProximityWarningTime > proximityWarningCooldown {
                                lastProximityWarningTime = currentTime
                                
                                // Get plate position
                                let position = determinePlatePosition(rect: rect, viewWidth: width)
                                
                                // Alert user about nearby license plate
                                let warningText = "Attention, license plate \(position) very close"
                                speakText(warningText)
                            }
                        }
                    }
                  } else {
                    boundingBoxViews[i].hide()
                  }
                }
              }
              // Write
              if developerMode {
                if save_detections {
                  saveText(text: str, file: "detections.txt")  // Write stats for each detection
                }
                if save_frames {
                  str = String(
                    format: "%.3f %.3f %.3f %.3f %.1f %.1f %.1f\n",
                    sec_day, freeSpace(), memoryUsage(), UIDevice.current.batteryLevel,
                    self.t1 * 1000, self.t2 * 1000, 1 / self.t4)
                  saveText(text: str, file: "frames.txt")  // Write stats for each image
                }
              }

              // Debug
              // print(str)
              // print(UIDevice.current.identifierForVendor!)
              // saveImage()
            }

          // Double tap işleme
            @objc func handleDoubleTap()
            {
                print("Double tap triggered") // Debug için log
                
                // Eğer zaten okuma yapılıyorsa, çık
                if isReading
                {
                    print("Already reading, exiting")
                    return
                }
                
                // Mevcut tespit edilen tüm plakaları kaydet
                platesQueue = [] // Temizle
                for i in 0..<boundingBoxViews.count
                {
                    if !boundingBoxViews[i].shapeLayer.isHidden
                    {
                        // Etiketi string olarak al
                        let boxLabel = boundingBoxViews[i].textLayer.string as? String ?? ""
                        print("Found box with label: \(boxLabel)")
                        
                        if boxLabel.lowercased().contains("license_plate")
                        {
                            print("Found license plate")
                            // Bounding box'ın şeklinden frame bilgisini al
                            if let path = boundingBoxViews[i].shapeLayer.path
                            {
                                let boxFrame = UIBezierPath(cgPath: path).bounds
                                platesQueue.append((boxFrame, boxLabel))
                                print("Added to queue: \(boxFrame), \(boxLabel)")
                            }
                        }
                    }
                }
        
        // Hiç plaka yoksa, bildir ve çık
        if platesQueue.isEmpty {
            print("No license plates found")
            speakText("No license plate detected")
            return
        }
        
        print("Starting to read \(platesQueue.count) plates")
        // Okuma işlemini başlat
        isReading = true
        readNextPlate()
    }

    // New function to provide directional guidance based on position
    // New function to provide directional guidance based on position
    func getDirectionalGuidance(for position: String) -> String {
        switch position {
        case "far left":
            return "Turn sharply to the left and proceed forward."
        case "on the left":
            return "Turn slightly to the left and proceed forward."
        case "directly in front":
            return "Proceed straight ahead."
        case "on the right":
            return "Turn slightly to the right and proceed forward."
        case "far right":
            return "Turn sharply to the right and proceed forward."
        default:
            return "Look around to locate it."
        }
    }
        // Improved plate position determination with more specific positions
        func determinePlatePosition(rect: CGRect, viewWidth: CGFloat) -> String {
            let centerX = rect.midX
            let viewFifth = viewWidth / 5
            
            if centerX < viewFifth {
                return "far left"
            } else if centerX < viewFifth * 2 {
                return "on the left"
            } else if centerX < viewFifth * 3 {
                return "directly in front"
            } else if centerX < viewFifth * 4 {
                return "on the right"
            } else {
                return "far right"
            }
        }

          // Sıradaki plakayı oku
          func readNextPlate() {
            if platesQueue.isEmpty {
              isReading = false
              return
            }
            
            let plate = platesQueue.removeFirst()
            let position = determinePlatePosition(rect: plate.rect, viewWidth: videoPreview.bounds.width)
            speakText("licence plate detected \(position)")
          }

          // Metni sesli oku
    
    // Fixed speech text function with consistent audio session
    func speakText(_ text: String) {
        print("Speaking with fixed audio session: \(text)")
        
        // Make sure we're not already speaking something
        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }
        
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        utterance.volume = 1.0
        
        // Configure audio session for playback
        do {
            // Use playback category for speaking which is more reliable
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
        } catch {
            print("Audio session error for speech: \(error)")
        }
        
        speechSynthesizer.delegate = self
        speechSynthesizer.speak(utterance)
    }


          // Pinch to Zoom Start ---------------------------------------------------------------------------------------------
          let minimumZoom: CGFloat = 1.0
          let maximumZoom: CGFloat = 10.0
          var lastZoomFactor: CGFloat = 1.0

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
          }  // Pinch to Zoom End --------------------------------------------------------------------------------------------
        }  // ViewController class End

        extension ViewController: VideoCaptureDelegate {
          func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
            predict(sampleBuffer: sampleBuffer)
          }
        }

        // Konuşma bittiğinde çağrılacak delegate
        extension ViewController: AVSpeechSynthesizerDelegate {
          func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
            // Bir sonraki plakayı oku
            readNextPlate()
          }
        }

        // Programmatically save image
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
              //
              //            // Save to camera roll
              //            UIImageWriteToSavedPhotosAlbum(img!, nil, nil, nil);
            } else {
              print("AVCapturePhotoCaptureDelegate Error")
            }
          }
        }
