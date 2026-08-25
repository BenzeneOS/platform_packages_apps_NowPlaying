def main [
    rust_tsv: path
    google_json_dir: path
    output_tsv: path
    --score-tolerance: float = 0.000022888
] {
    let comparison = (open $rust_tsv | each { |rust|
        let capture = ($rust.capture | path basename | path parse | get stem)
        let google = (open ($google_json_dir | path join $"($capture).json"))
        let accepted_candidates = ($google.candidates | where accepted == true)
        let google_accepted = (($accepted_candidates | length) > 0)
        let google_match = if $google_accepted { $accepted_candidates | first } else { null }
        let rust_accepted = ($rust.accepted == 1)
        let acceptance_agree = ($rust_accepted == $google_accepted)
        let identity_agree = if ($rust_accepted and $google_accepted) {
            $rust.track == $google_match.track.media_id
        } else {
            $acceptance_agree
        }
        let score_delta = if ($rust_accepted and $google_accepted and $identity_agree) {
            (($rust.score - $google_match.score) | math abs)
        } else {
            ""
        }
        {
            capture: $capture
            rust_accepted: $rust_accepted
            google_accepted: $google_accepted
            rust_track: $rust.track
            google_track: (if $google_accepted { $google_match.track.media_id } else { "" })
            rust_numeric_id: $rust.numeric_id
            google_numeric_id: (if $google_accepted { $google_match.numeric_id } else { "" })
            rust_shard: $rust.shard
            google_shard: (if $google_accepted { $google_match.shard } else { "" })
            rust_offset: $rust.offset
            google_offset: (if $google_accepted { $google_match.offset_seconds } else { "" })
            offset_delta: (if ($rust_accepted and $google_accepted) {
                (($rust.offset - ($google_match.offset_seconds | into int)) | math abs)
            } else {
                ""
            })
            rust_score: $rust.score
            google_score: (if $google_accepted { $google_match.score } else { "" })
            score_delta: $score_delta
            score_within_tolerance: (if ($score_delta | is-empty) {
                true
            } else {
                $score_delta <= $score_tolerance
            })
            acceptance_agree: $acceptance_agree
            identity_agree: $identity_agree
        }
    })
    $comparison | to tsv | save --force $output_tsv
    let required_disagreements = ($comparison | where acceptance_agree == false or identity_agree == false)
    let score_disagreements = ($comparison | where score_within_tolerance == false)
    print $"captures\t($comparison | length)"
    print $"required_disagreements\t($required_disagreements | length)"
    print $"score_disagreements\t($score_disagreements | length)"
    if (($required_disagreements | length) > 0) {
        error make {msg: "acceptance or track identity disagreement"}
    }
}
