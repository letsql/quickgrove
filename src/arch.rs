#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct CpuFeatures {
    has_prefetch: bool,
}
impl Default for CpuFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuFeatures {
    #[inline]
    pub fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let has_prefetch = is_x86_feature_detected!("sse");
            Self { has_prefetch }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                has_prefetch: false,
            }
        }
    }

    #[inline]
    pub fn prefetch<T>(&self, _ptr: *const T) {
        if self.has_prefetch {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                _mm_prefetch(_ptr as *const i8, _MM_HINT_T0);
            }
        }
    }
}
