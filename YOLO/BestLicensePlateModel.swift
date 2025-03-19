//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//  This file is part of the Ultralytics YOLO app for license plate detection

import Foundation
import CoreML

class BestLicensePlateModel {
    // Model property
    private var _model: MLModel
    
    // Use a computed property for public access
    var model: MLModel {
        return _model
    }
    
    public init(configuration: MLModelConfiguration) throws {
        let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlmodelc")!
        self._model = try MLModel(contentsOf: modelURL, configuration: configuration)
    }
    
    public convenience init() throws {
        try self.init(configuration: MLModelConfiguration())
    }
}
