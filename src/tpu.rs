use crate::camera::CameraFrame;
use crate::detector::Detection;
use std::error::Error;
use std::fmt;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct InferenceConfig {
    pub classes_num: i32,
    pub confidence_threshold: f32,
    pub iou_threshold: f32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            classes_num: 1,
            confidence_threshold: 0.5,
            iou_threshold: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct InferTiming {
    /// JPEG decode microseconds.
    pub decode_us: i64,
    /// Resize, letterbox clear, and planar pack microseconds.
    pub resize_us: i64,
    /// total preprocess microseconds.
    pub preprocess_us: i64,
    /// CVI_NN_Forward microseconds.
    pub forward_us: i64,
    /// detection parse + dequant + NMS + box correction microseconds.
    pub postprocess_us: i64,
}

#[derive(Debug)]
pub struct TpuError(String);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelInputFormat {
    RgbPlanar,
    Yuv420Planar,
    YuvNv12,
    YuvNv21,
}

impl ModelInputFormat {
    pub fn buffer_len(self, width: i32, height: i32) -> Result<usize, TpuError> {
        if width <= 0 || height <= 0 {
            return Err(TpuError::new("input tensor dimensions must be positive"));
        }
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| TpuError::new("input tensor dimensions overflow"))?;
        match self {
            Self::RgbPlanar => pixels
                .checked_mul(3)
                .ok_or_else(|| TpuError::new("RGB input tensor dimensions overflow")),
            Self::Yuv420Planar | Self::YuvNv12 | Self::YuvNv21 => pixels
                .checked_mul(3)
                .and_then(|value| value.checked_div(2))
                .ok_or_else(|| TpuError::new("YUV420 input tensor dimensions overflow")),
        }
    }
}

impl TpuError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for TpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for TpuError {}

#[cfg(target_arch = "riscv64")]
mod imp {
    use super::{CameraFrame, Detection, InferTiming, InferenceConfig, ModelInputFormat, TpuError};
    use crate::detector::{correct_yolo_boxes, nms, parse_yolov8_output};
    use crate::image_bridge;
    use std::ffi::{c_char, c_int, c_void, CString};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::ptr;
    use std::slice;
    use std::time::Instant;

    const CVI_FMT_FP32: i32 = 0;
    const CVI_FMT_BF16: i32 = 3;
    const CVI_FMT_INT16: i32 = 4;
    const CVI_FMT_INT8: i32 = 6;
    const CVI_FMT_UINT8: i32 = 7;
    const CVI_RC_SUCCESS: i32 = 0;
    const CVI_DIM_MAX: usize = 6;
    const CVI_NN_PIXEL_RGB_PLANAR: i32 = 2;
    const CVI_NN_PIXEL_YUV_NV12: i32 = 11;
    const CVI_NN_PIXEL_YUV_NV21: i32 = 12;
    const CVI_NN_PIXEL_YUV_420_PLANAR: i32 = 13;

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct CviShape {
        dim: [i32; CVI_DIM_MAX],
        dim_size: usize,
    }

    #[repr(C)]
    #[derive(Debug)]
    struct CviTensor {
        name: *mut c_char,
        shape: CviShape,
        fmt: i32,
        count: usize,
        mem_size: usize,
        sys_mem: *mut u8,
        paddr: u64,
        mem_type: i32,
        qscale: f32,
        zero_point: c_int,
        pixel_format: i32,
        aligned: bool,
        mean: [f32; 3],
        scale: [f32; 3],
        owner: *mut c_void,
        reserved: [c_char; 32],
    }

    type CviModelHandle = *mut c_void;

    unsafe extern "C" {
        fn CVI_NN_RegisterModel(model_file: *const c_char, model: *mut CviModelHandle) -> i32;
        fn CVI_NN_GetInputOutputTensors(
            model: CviModelHandle,
            inputs: *mut *mut CviTensor,
            input_num: *mut i32,
            outputs: *mut *mut CviTensor,
            output_num: *mut i32,
        ) -> i32;
        fn CVI_NN_GetTensorByName(
            name: *const c_char,
            tensors: *mut CviTensor,
            num: i32,
        ) -> *mut CviTensor;
        fn CVI_NN_TensorPtr(tensor: *mut CviTensor) -> *mut c_void;
        fn CVI_NN_TensorShape(tensor: *mut CviTensor) -> CviShape;
        fn CVI_NN_Forward(
            model: CviModelHandle,
            inputs: *mut CviTensor,
            input_num: i32,
            outputs: *mut CviTensor,
            output_num: i32,
        ) -> i32;
        fn CVI_NN_CleanupModel(model: CviModelHandle) -> i32;
    }

    pub struct YoloModel {
        model: CviModelHandle,
        inputs: *mut CviTensor,
        input_num: i32,
        outputs: *mut CviTensor,
        output_num: i32,
        input: *mut CviTensor,
        input_h: i32,
        input_w: i32,
        input_format: ModelInputFormat,
        output_shapes: Vec<CviShape>,
        preprocessor: image_bridge::ImagePreprocessor,
    }

    // The model handle is owned by a single TPU worker thread. We do not share
    // it across threads; we only move ownership into the worker.
    unsafe impl Send for YoloModel {}

    impl YoloModel {
        pub fn open(path: &Path) -> Result<Self, TpuError> {
            let c_path = CString::new(path.as_os_str().as_bytes())
                .map_err(|_| TpuError::new("model path contains NUL byte"))?;
            let mut model: CviModelHandle = ptr::null_mut();
            let rc = unsafe { CVI_NN_RegisterModel(c_path.as_ptr(), &mut model) };
            if rc != CVI_RC_SUCCESS {
                return Err(TpuError::new(format!("CVI_NN_RegisterModel failed: {rc}")));
            }

            let mut inputs = ptr::null_mut();
            let mut outputs = ptr::null_mut();
            let mut input_num = 0;
            let mut output_num = 0;
            let rc = unsafe {
                CVI_NN_GetInputOutputTensors(
                    model,
                    &mut inputs,
                    &mut input_num,
                    &mut outputs,
                    &mut output_num,
                )
            };
            if rc != CVI_RC_SUCCESS {
                unsafe {
                    CVI_NN_CleanupModel(model);
                }
                return Err(TpuError::new(format!(
                    "CVI_NN_GetInputOutputTensors failed: {rc}"
                )));
            }

            let input = unsafe { CVI_NN_GetTensorByName(ptr::null(), inputs, input_num) };
            if input.is_null() {
                unsafe {
                    CVI_NN_CleanupModel(model);
                }
                return Err(TpuError::new("default input tensor not found"));
            }

            let input_shape = unsafe { CVI_NN_TensorShape(input) };
            let input_h = input_shape.dim[2];
            let input_w = input_shape.dim[3];
            let input_tensor = unsafe { &*input };
            let input_format = match input_tensor.pixel_format {
                CVI_NN_PIXEL_YUV_420_PLANAR => ModelInputFormat::Yuv420Planar,
                CVI_NN_PIXEL_YUV_NV12 => ModelInputFormat::YuvNv12,
                CVI_NN_PIXEL_YUV_NV21 => ModelInputFormat::YuvNv21,
                CVI_NN_PIXEL_RGB_PLANAR => ModelInputFormat::RgbPlanar,
                _ => ModelInputFormat::RgbPlanar,
            };
            let outputs_slice = unsafe { slice::from_raw_parts_mut(outputs, output_num as usize) };
            let output_shapes = outputs_slice
                .iter_mut()
                .map(|tensor| unsafe { CVI_NN_TensorShape(tensor as *mut CviTensor) })
                .collect();

            Ok(Self {
                model,
                inputs,
                input_num,
                outputs,
                output_num,
                input,
                input_h,
                input_w,
                input_format,
                output_shapes,
                preprocessor: image_bridge::ImagePreprocessor::new(),
            })
        }

        pub fn infer(
            &mut self,
            frame: &CameraFrame,
            config: InferenceConfig,
        ) -> Result<Vec<Detection>, TpuError> {
            self.infer_timed(frame, config, None)
        }

        pub fn input_dimensions(&self) -> (i32, i32) {
            (self.input_w, self.input_h)
        }

        pub fn input_format(&self) -> ModelInputFormat {
            self.input_format
        }

        pub fn infer_timed(
            &mut self,
            frame: &CameraFrame,
            config: InferenceConfig,
            mut timing: Option<&mut InferTiming>,
        ) -> Result<Vec<Detection>, TpuError> {
            let (input_ptr, input_len) = self.input_buffer()?;
            let input = unsafe { slice::from_raw_parts_mut(input_ptr, input_len) };

            // Camera already delivers I422 planar at the model input size, so
            // this is a plain color convert into the RGB planar tensor: no JPEG
            // decode, no resize, no letterbox.
            let pre_start = Instant::now();
            image_bridge::yuv422p_to_rgb_planar(&frame.pixels, input, self.input_w, self.input_h)
                .map_err(|err| TpuError::new(format!("YUV preprocess failed: {err}")))?;
            let preprocess_us = pre_start.elapsed().as_micros() as i64;

            let (detections, forward_us, postprocess_us) =
                self.forward_and_detections(config, frame.width as i32, frame.height as i32)?;

            if let Some(t) = timing.as_deref_mut() {
                *t = InferTiming {
                    decode_us: 0,
                    resize_us: preprocess_us,
                    preprocess_us,
                    forward_us,
                    postprocess_us,
                };
            }
            Ok(detections)
        }

        pub fn infer_rgb_planar_timed(
            &mut self,
            rgb_planar: &[u8],
            image_w: i32,
            image_h: i32,
            config: InferenceConfig,
            mut timing: Option<&mut InferTiming>,
        ) -> Result<Vec<Detection>, TpuError> {
            let copy_start = Instant::now();
            let (input_ptr, input_len) = self.input_buffer()?;
            if rgb_planar.len() < input_len {
                return Err(TpuError::new("RGB planar input buffer is too small"));
            }
            let input = unsafe { slice::from_raw_parts_mut(input_ptr, input_len) };
            input.copy_from_slice(&rgb_planar[..input_len]);
            let copy_us = copy_start.elapsed().as_micros() as i64;

            let (detections, forward_us, postprocess_us) =
                self.forward_and_detections(config, image_w, image_h)?;

            if let Some(t) = timing.as_deref_mut() {
                *t = InferTiming {
                    decode_us: 0,
                    resize_us: copy_us,
                    preprocess_us: copy_us,
                    forward_us,
                    postprocess_us,
                };
            }
            Ok(detections)
        }

        pub fn infer_yuv420_timed(
            &mut self,
            yuv420: &[u8],
            image_w: i32,
            image_h: i32,
            config: InferenceConfig,
            mut timing: Option<&mut InferTiming>,
        ) -> Result<Vec<Detection>, TpuError> {
            if !matches!(
                self.input_format,
                ModelInputFormat::Yuv420Planar
                    | ModelInputFormat::YuvNv12
                    | ModelInputFormat::YuvNv21
            ) {
                return Err(TpuError::new("model input is not YUV420"));
            }

            let copy_start = Instant::now();
            let (input_ptr, input_len) = self.input_buffer()?;
            if yuv420.len() < input_len {
                return Err(TpuError::new("YUV420 input buffer is too small"));
            }
            let input = unsafe { slice::from_raw_parts_mut(input_ptr, input_len) };
            input.copy_from_slice(&yuv420[..input_len]);
            let copy_us = copy_start.elapsed().as_micros() as i64;

            let (detections, forward_us, postprocess_us) =
                self.forward_and_detections(config, image_w, image_h)?;

            if let Some(t) = timing.as_deref_mut() {
                *t = InferTiming {
                    decode_us: 0,
                    resize_us: copy_us,
                    preprocess_us: copy_us,
                    forward_us,
                    postprocess_us,
                };
            }
            Ok(detections)
        }

        /// Resolve the input tensor's backing buffer as a raw pointer and its
        /// required input length. Returned as a raw pointer (not a slice)
        /// so callers can also borrow `self` afterwards.
        fn input_buffer(&self) -> Result<(*mut u8, usize), TpuError> {
            let input_ptr = unsafe { CVI_NN_TensorPtr(self.input) as *mut u8 };
            if input_ptr.is_null() {
                return Err(TpuError::new("input tensor pointer is null"));
            }
            let input_len = self.input_format.buffer_len(self.input_w, self.input_h)?;
            let input_tensor = unsafe { &*self.input };
            if input_tensor.mem_size < input_len {
                return Err(TpuError::new(format!(
                    "input tensor buffer is too small: mem_size={} required={input_len}",
                    input_tensor.mem_size
                )));
            }
            Ok((input_ptr, input_len))
        }

        /// Run the network on the already-populated input tensor and turn the
        /// output into detections. `image_w`/`image_h` are the original frame
        /// dimensions used to map boxes back out of the letterboxed input.
        /// Returns the detections plus forward/postprocess timings (µs).
        fn forward_and_detections(
            &mut self,
            config: InferenceConfig,
            image_w: i32,
            image_h: i32,
        ) -> Result<(Vec<Detection>, i64, i64), TpuError> {
            let fwd_start = Instant::now();
            let rc = unsafe {
                CVI_NN_Forward(
                    self.model,
                    self.inputs,
                    self.input_num,
                    self.outputs,
                    self.output_num,
                )
            };
            let forward_us = fwd_start.elapsed().as_micros() as i64;
            if rc != CVI_RC_SUCCESS {
                return Err(TpuError::new(format!("CVI_NN_Forward failed: {rc}")));
            }

            let post_start = Instant::now();
            let mut detections = self.get_detections(config)?;
            nms(&mut detections, config.iou_threshold);
            correct_yolo_boxes(
                &mut detections,
                image_h,
                image_w,
                self.input_h,
                self.input_w,
            );
            let postprocess_us = post_start.elapsed().as_micros() as i64;

            Ok((detections, forward_us, postprocess_us))
        }

        /// Run inference on a standalone image (JPEG/PNG/...) and write a copy
        /// with the detection boxes drawn to out_path. Unlike the camera hot
        /// path this decodes and letterboxes the image on the CPU.
        pub fn detect_image(
            &mut self,
            image: &[u8],
            out_path: &Path,
            config: InferenceConfig,
        ) -> Result<Vec<Detection>, TpuError> {
            let (input_ptr, input_len) = self.input_buffer()?;
            let input = unsafe { slice::from_raw_parts_mut(input_ptr, input_len) };

            let preprocess = self
                .preprocessor
                .mjpeg_to_rgb_planar(image, input, self.input_w, self.input_h)
                .map_err(|err| TpuError::new(format!("MJPEG decode/preprocess failed: {err}")))?;

            let image_w = if preprocess.src_w > 0 {
                preprocess.src_w
            } else {
                self.input_w
            };
            let image_h = if preprocess.src_h > 0 {
                preprocess.src_h
            } else {
                self.input_h
            };
            let (detections, _, _) = self.forward_and_detections(config, image_w, image_h)?;

            image_bridge::draw_detections(image, &detections, out_path)
                .map_err(|err| TpuError::new(format!("failed to write annotated image: {err}")))?;
            Ok(detections)
        }

        fn get_detections(&mut self, config: InferenceConfig) -> Result<Vec<Detection>, TpuError> {
            if self.output_num < 1 || self.output_shapes.is_empty() {
                return Err(TpuError::new("model has no output tensor"));
            }
            let output = unsafe { &mut *self.outputs };
            let shape = self.output_shapes[0];
            let count = output.count;
            let ptr = unsafe { CVI_NN_TensorPtr(output as *mut CviTensor) };
            if ptr.is_null() {
                return Err(TpuError::new("output tensor pointer is null"));
            }

            let data = tensor_to_f32(output, ptr, count)?;
            Ok(parse_yolov8_output(
                &data,
                [shape.dim[0], shape.dim[1], shape.dim[2], shape.dim[3]],
                config.classes_num,
                config.confidence_threshold,
            ))
        }
    }

    impl Drop for YoloModel {
        fn drop(&mut self) {
            if !self.model.is_null() {
                unsafe {
                    CVI_NN_CleanupModel(self.model);
                }
            }
        }
    }

    fn tensor_to_f32(
        tensor: &CviTensor,
        ptr: *mut c_void,
        count: usize,
    ) -> Result<Vec<f32>, TpuError> {
        match tensor.fmt {
            CVI_FMT_FP32 => {
                let src = unsafe { slice::from_raw_parts(ptr as *const f32, count) };
                Ok(src.to_vec())
            }
            CVI_FMT_INT8 => {
                let src = unsafe { slice::from_raw_parts(ptr as *const i8, count) };
                Ok(src.iter().map(|v| *v as f32 * tensor.qscale).collect())
            }
            CVI_FMT_UINT8 => {
                let src = unsafe { slice::from_raw_parts(ptr as *const u8, count) };
                Ok(src
                    .iter()
                    .map(|v| (*v as i32 - tensor.zero_point) as f32 * tensor.qscale)
                    .collect())
            }
            CVI_FMT_BF16 => {
                let src = unsafe { slice::from_raw_parts(ptr as *const u16, count) };
                Ok(src
                    .iter()
                    .map(|v| f32::from_bits((*v as u32) << 16))
                    .collect())
            }
            CVI_FMT_INT16 => {
                let src = unsafe { slice::from_raw_parts(ptr as *const i16, count) };
                Ok(src.iter().map(|v| *v as f32 * tensor.qscale).collect())
            }
            other => Err(TpuError::new(format!(
                "unsupported output tensor format: {other}"
            ))),
        }
    }
}

#[cfg(not(target_arch = "riscv64"))]
mod imp {
    use super::{CameraFrame, Detection, InferTiming, InferenceConfig, ModelInputFormat, TpuError};
    use std::path::Path;

    pub struct YoloModel;

    impl YoloModel {
        pub fn open(_path: &Path) -> Result<Self, TpuError> {
            Err(TpuError::new(
                "akars was built without SG2002 TPU runtime support",
            ))
        }

        pub fn infer(
            &mut self,
            _frame: &CameraFrame,
            _config: InferenceConfig,
        ) -> Result<Vec<Detection>, TpuError> {
            Err(TpuError::new(
                "akars was built without SG2002 TPU runtime support",
            ))
        }

        pub fn input_dimensions(&self) -> (i32, i32) {
            (0, 0)
        }

        pub fn input_format(&self) -> ModelInputFormat {
            ModelInputFormat::RgbPlanar
        }

        pub fn infer_timed(
            &mut self,
            _frame: &CameraFrame,
            _config: InferenceConfig,
            _timing: Option<&mut InferTiming>,
        ) -> Result<Vec<Detection>, TpuError> {
            Err(TpuError::new(
                "akars was built without SG2002 TPU runtime support",
            ))
        }

        pub fn infer_rgb_planar_timed(
            &mut self,
            _rgb_planar: &[u8],
            _image_w: i32,
            _image_h: i32,
            _config: InferenceConfig,
            _timing: Option<&mut InferTiming>,
        ) -> Result<Vec<Detection>, TpuError> {
            Err(TpuError::new(
                "akars was built without SG2002 TPU runtime support",
            ))
        }

        pub fn infer_yuv420_timed(
            &mut self,
            _yuv420: &[u8],
            _image_w: i32,
            _image_h: i32,
            _config: InferenceConfig,
            _timing: Option<&mut InferTiming>,
        ) -> Result<Vec<Detection>, TpuError> {
            Err(TpuError::new(
                "akars was built without SG2002 TPU runtime support",
            ))
        }

        pub fn detect_image(
            &mut self,
            _image: &[u8],
            _out_path: &Path,
            _config: InferenceConfig,
        ) -> Result<Vec<Detection>, TpuError> {
            Err(TpuError::new(
                "akars was built without SG2002 TPU runtime support",
            ))
        }
    }
}

pub use imp::YoloModel;

pub fn open_model(path: impl AsRef<Path>) -> Result<YoloModel, TpuError> {
    YoloModel::open(path.as_ref())
}
