pragma circom 2.1.6;

include "../../../node_modules/circomlib/circuits/comparators.circom";

template DiabetesRiskAlert() {
    // Public inputs
    signal input glucose_threshold;
    signal input risk_threshold;
    signal input result_commitment;

    // Private inputs
    signal input glucose_reading;
    signal input risk_score;
    signal input witness_randomness;

    // Output
    signal output alert_status;

    // Check if glucose > glucose_threshold
    component glucose_gt = GreaterThan(10); // 10 bits for glucose values
    glucose_gt.in[0] <== glucose_reading;
    glucose_gt.in[1] <== glucose_threshold;

    // Check if risk_score > risk_threshold (scaled to integers)
    signal risk_score_scaled;
    signal risk_threshold_scaled;
    risk_score_scaled <== risk_score * 1000; // Scale to avoid decimals
    risk_threshold_scaled <== risk_threshold * 1000;

    component risk_gt = GreaterThan(10);
    risk_gt.in[0] <== risk_score_scaled;
    risk_gt.in[1] <== risk_threshold_scaled;

    // AND condition: both must be true for alert
    alert_status <== glucose_gt.out * risk_gt.out;

    // Verify commitment (simplified)
    signal computed_commitment;
    computed_commitment <== alert_status + witness_randomness;

    // In production, would verify the commitment matches
    component commitment_check = IsEqual();
    commitment_check.in[0] <== computed_commitment;
    commitment_check.in[1] <== result_commitment;
}

component main {public [glucose_threshold, risk_threshold, result_commitment]} = DiabetesRiskAlert();
