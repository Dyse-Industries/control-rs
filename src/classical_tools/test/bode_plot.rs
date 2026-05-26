// use std::ops::Neg;
//
// #[cfg(feature = "std")]
// /// Renders a single magnitude and phase plot on subplots
// fn render_bode_subplot<T>(
//     plot: &mut plotly::Plot,
//     frequencies: &[T],
//     mag: &[T],
//     phase: &[T],
//     row: usize,
//     col: usize,
//     margins: &PhaseGainCrossover<T>,
// ) where
//     T: 'static
//     + Copy
//     + Neg<Output = T>
//     + Zero
//     + One
//     + Magnitude
//     + Phase
//     + serde::ser::Serialize,
// {
//     use plotly::common;
//     extern crate std;
//     use std::{format, vec, vec::Vec};
//
//     let mag_db: Vec<T> = mag.iter().map(|&m| m.to_db()).collect();
//     let phase_deg: Vec<T> = phase.iter().map(|&p| p.to_degrees()).collect();
//
//     // Add magnitude plot
//     plot.add_trace(
//         plotly::Scatter::new(frequencies.to_vec(), mag_db)
//             .mode(common::Mode::Lines)
//             .name(format!("Magnitude[{row}, {col}]"))
//             // .x_axis(x_axis_mag.clone())
//             // .y_axis(y_axis_mag.clone()).color(Rgb::new(0, 0, 255))
//             .marker(common::Marker::new()),
//     );
//
//     // Add phase plot
//     plot.add_trace(
//         plotly::Scatter::new(frequencies.to_vec(), phase_deg)
//             .mode(common::Mode::Lines)
//             .name(format!("Phase[{row}, {col}]"))
//             .x_axis("x2")
//             .y_axis("y2")
//             .marker(common::Marker::new()),
//     );
//
//     // Gain margin line
//     if let (Some(wc), Some(gm)) = (margins.phase_crossover, margins.gain_margin)
//     {
//         plot.add_trace(
//             plotly::Scatter::new(vec![wc, wc], vec![-gm, T::zero()])
//                 .mode(common::Mode::Lines)
//                 .name(format!("Gain FrequencyMargin[{row}, {col}]"))
//                 // .x_axis(x_axis_mag)
//                 // .y_axis(y_axis_mag)
//                 .line(
//                     common::Line::new()
//                         .dash(common::DashType::Dot)
//                         .color(plotly::color::Rgb::new(0, 0, 0)),
//                 ),
//         );
//     }
//
//     // Phase margin line
//     if let (Some(wc), Some(pm)) = (margins.gain_crossover, margins.phase_margin)
//     {
//         plot.add_trace(
//             plotly::Scatter::new(
//                 vec![wc, wc],
//                 vec![T::ONE_EIGHTY.neg(), T::ONE_EIGHTY.neg() + pm],
//             )
//                 .mode(common::Mode::Lines)
//                 .name(format!("Phase FrequencyMargin[{row}, {col}]"))
//                 .x_axis("x2")
//                 .y_axis("y2")
//                 .line(
//                     common::Line::new()
//                         .dash(common::DashType::Dot)
//                         .color(plotly::color::Rgb::new(0, 0, 0)),
//                 ),
//         );
//     }
// }
//
// #[cfg(feature = "std")]
// /// Renders a Bode plot for an object implementing `FrequencyTools`
// pub fn bode<T, F, const N: usize, const M: usize, const K: usize>(
//     title: &str,
//     system: &F,
//     mut response: FrequencyResponse<T, N, M, K>,
// ) -> plotly::Plot
// where
//     T: Copy
//     + Neg<Output = T>
//     + Zero
//     + One
//     + Real
//     + Magnitude
//     + Phase
//     + serde::ser::Serialize,
//     F: FrequencyTools<T, N, M>,
// {
//     use plotly::Layout;
//
//     if response.responses.is_none() {
//         system.frequency_response(&mut response);
//     }
//     let margins = FrequencyMargin::new(&response);
//
//     let mut plot = plotly::Plot::new();
//     plot.set_layout(
//         Layout::new()
//             .title(plotly::common::Title::with_text(title))
//             .x_axis(
//                 plotly::layout::Axis::new()
//                     .title(plotly::common::Title::with_text(
//                         "Frequency (rad/s)",
//                     ))
//                     .type_(plotly::layout::AxisType::Log),
//             )
//             .y_axis(
//                 plotly::layout::Axis::new()
//                     .title(plotly::common::Title::with_text("Magnitude (dB)")),
//             )
//             .x_axis2(
//                 plotly::layout::Axis::new()
//                     .title(plotly::common::Title::with_text(
//                         "Frequency (rad/s)",
//                     ))
//                     .type_(plotly::layout::AxisType::Log),
//             )
//             .y_axis2(
//                 plotly::layout::Axis::new()
//                     .title(plotly::common::Title::with_text("Phase (deg)")),
//             )
//             .grid(
//                 plotly::layout::LayoutGrid::new()
//                     .rows(2)
//                     .columns(1)
//                     .pattern(plotly::layout::GridPattern::Independent)
//                     .row_order(plotly::layout::RowOrder::TopToBottom),
//             ),
//     );
//
//     // extract each output mag/phase (make a helper in response soon)
//     for in_channel in 0..N {
//         for out_channel in 0..M {
//             if let Some((magnitudes, phases)) =
//                 response.mag_phase(in_channel, out_channel)
//             {
//                 // render the output as the response to each input
//                 render_bode_subplot(
//                     &mut plot,
//                     &response.frequencies,
//                     &magnitudes,
//                     &phases,
//                     in_channel,
//                     out_channel,
//                     &margins.0[in_channel][out_channel],
//                 );
//             }
//         }
//     }
//
//     plot
// }