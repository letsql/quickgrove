#[derive(Debug, Clone)]
pub enum Objective {
    SquaredError,
}

impl Objective {
    #[inline(always)]
    pub fn compute_score(&self, leaf_weight: f32) -> f32 {
        match self {
            Objective::SquaredError => leaf_weight,
        }
    }
}
