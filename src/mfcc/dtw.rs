use std::cmp;

pub struct Dtw<T: Copy> {
    state_m: usize,
    state_n: usize,
    distance_fn: fn(T, T) -> f32,
    state_similarity: Option<f32>,
    distance_cost_matrix: Option<Vec<Vec<f32>>>,
}
impl<T: Copy> Dtw<T> {
    pub fn compute_optimal_path(&mut self, first_sequence: &[T], second_sequence: &[T]) -> f32 {
        self.state_m = first_sequence.len();
        self.state_n = second_sequence.len();
        let mut distance_cost_matrix: Vec<Vec<f32>> = infinity_matrix(self.state_m, self.state_n);
        distance_cost_matrix[0][0] = (self.distance_fn)(first_sequence[0], second_sequence[0]);
        for (row_index, first_sequence_item) in
            first_sequence.iter().enumerate().take(self.state_m).skip(1)
        {
            let cost = (self.distance_fn)(*first_sequence_item, second_sequence[0]);
            distance_cost_matrix[row_index][0] = cost + distance_cost_matrix[row_index - 1][0];
        }
        for (column_index, second_sequence_item) in second_sequence
            .iter()
            .enumerate()
            .take(self.state_n)
            .skip(1)
        {
            let cost = (self.distance_fn)(first_sequence[0], *second_sequence_item);
            distance_cost_matrix[0][column_index] =
                cost + distance_cost_matrix[0][column_index - 1];
        }
        for (row_index, first_sequence_item) in
            first_sequence.iter().enumerate().take(self.state_m).skip(1)
        {
            for (column_index, second_sequence_item) in second_sequence
                .iter()
                .enumerate()
                .take(self.state_n)
                .skip(1)
            {
                let cost = (self.distance_fn)(*first_sequence_item, *second_sequence_item);
                let insertion = distance_cost_matrix[row_index - 1][column_index];
                let deletion = distance_cost_matrix[row_index][column_index - 1];
                let matches = distance_cost_matrix[row_index - 1][column_index - 1];
                let min_value = [insertion, deletion, matches]
                    .iter()
                    .fold(f32::INFINITY, |a, &b| a.min(b));
                distance_cost_matrix[row_index][column_index] = cost + min_value;
            }
        }
        let similarity = distance_cost_matrix[self.state_m - 1][self.state_n - 1];
        self.distance_cost_matrix = Option::Some(distance_cost_matrix);
        self.state_similarity = Option::Some(similarity);
        similarity
    }
    pub fn compute_optimal_path_with_window(
        &mut self,
        first_sequence: &[T],
        second_sequence: &[T],
        w: u16,
    ) -> f32 {
        self.state_m = first_sequence.len();
        self.state_n = second_sequence.len();
        let window = cmp::max(
            w as usize,
            abs_diff(first_sequence.len(), second_sequence.len()),
        );

        let mut distance_cost_matrix: Vec<Vec<f32>> =
            infinity_matrix(self.state_m + 1, self.state_n + 1);
        distance_cost_matrix[0][0] = 0.;
        for row_index in 1..=self.state_m {
            let start_index = if row_index > window {
                cmp::max(1, row_index - window)
            } else {
                1
            };
            for column_index in start_index..cmp::min(self.state_n + 1, row_index + window) {
                let cost = (self.distance_fn)(
                    first_sequence[row_index - 1],
                    second_sequence[column_index - 1],
                );
                let insertion = distance_cost_matrix[row_index - 1][column_index];
                let deletion = distance_cost_matrix[row_index][column_index - 1];
                let matches = distance_cost_matrix[row_index - 1][column_index - 1];
                let min_value = [insertion, deletion, matches]
                    .iter()
                    .fold(f32::INFINITY, |a, &b| a.min(b));
                distance_cost_matrix[row_index][column_index] = cost + min_value;
            }
        }
        // resize matrix
        let mut final_distance_cost_matrix: Vec<Vec<f32>> =
            infinity_matrix(self.state_m + 1, self.state_n);
        for row_index in 0..=self.state_m {
            for col_index in 1..=self.state_n {
                final_distance_cost_matrix[row_index][col_index - 1] =
                    distance_cost_matrix[row_index][col_index];
            }
        }
        let similarity = final_distance_cost_matrix[self.state_m - 1][self.state_n - 1];
        self.distance_cost_matrix = Option::Some(final_distance_cost_matrix);
        self.state_similarity = Option::Some(similarity);
        similarity
    }
    pub fn retrieve_optimal_path(&self) -> Option<Vec<[usize; 2]>> {
        self.distance_cost_matrix.as_ref()?;
        let distance_cost_matrix = self.distance_cost_matrix.as_ref().unwrap();
        let mut row_index = self.state_m - 1;
        let mut column_index = self.state_n - 1;
        let mut path = vec![[0usize; 2]; cmp::min(row_index, column_index)];
        while (row_index > 0) || (column_index > 0) {
            if row_index > 0 && column_index > 0 {
                let insertion = distance_cost_matrix[row_index - 1][column_index];
                let deletion = distance_cost_matrix[row_index][column_index - 1];
                let matches = distance_cost_matrix[row_index - 1][column_index - 1];
                let min_value = [insertion, deletion, matches]
                    .iter()
                    .fold(f32::INFINITY, |a, &b| a.min(b));
                if min_value == matches {
                    row_index -= 1;
                    column_index -= 1;
                } else if min_value == insertion {
                    row_index -= 1;
                } else if min_value == deletion {
                    column_index -= 1;
                }
            } else if row_index > 0 && column_index == 0 {
                row_index -= 1;
            } else if row_index == 0 && column_index > 0 {
                column_index -= 1;
            }
            let part = [row_index, column_index];
            path.push(part);
        }
        path.reverse();
        Some(path)
    }
    pub fn new(distance_fn: fn(T, T) -> f32) -> Dtw<T> {
        Dtw {
            state_m: 0,
            state_n: 0,
            distance_fn,
            state_similarity: Option::None,
            distance_cost_matrix: Option::None,
        }
    }
}
fn infinity_matrix(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    vec![vec![f32::INFINITY; cols]; rows]
}
fn abs_diff(a: usize, b: usize) -> usize {
    if a > b {
        a - b
    } else {
        b - a
    }
}
