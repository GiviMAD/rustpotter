use super::{dtw::Dtw, MfccComparator};

pub struct MfccAverager {}
impl MfccAverager {
    pub fn average(mut other_mfccs: Vec<Vec<Vec<f32>>>) -> Option<Vec<Vec<f32>>> {
        let mut origin = other_mfccs.drain(0..1).next().unwrap();
        for frames in other_mfccs.iter() {
            let mut dtw = Dtw::new(MfccComparator::calculate_distance);
            dtw.compute_optimal_path(
                &origin.iter().map(|item| &item[..]).collect::<Vec<_>>(),
                &frames.iter().map(|item| &item[..]).collect::<Vec<_>>(),
            );
            let mut avgs = origin
                .iter()
                .map(|x| x.iter().map(|&y| vec![y]).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            dtw.retrieve_optimal_path()
                .unwrap()
                .into_iter()
                .for_each(|[x, y]| {
                    frames[y].iter().enumerate().for_each(|(index, feature)| {
                        avgs[x][index].push(*feature);
                    })
                });
            origin = avgs
                .iter()
                .map(|x| {
                    x.iter()
                        .map(|feature_group| {
                            feature_group.to_vec().iter().sum::<f32>() / feature_group.len() as f32
                        })
                        .collect::<Vec<f32>>()
                })
                .collect::<Vec<_>>();
        }
        Some(origin)
    }
}
