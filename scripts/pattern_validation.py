#!/usr/bin/env python3
"""
Pattern Validation Script

Implements the Learning Pattern Validation Prompt for adaptive safety learning systems.
"""

import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time
import numpy as np

@dataclass
class PatternData:
    pattern_id: str
    features: List[float]
    threshold: float
    confidence: float
    usage_count: int
    success_rate: float
    creation_time: float
    evolution_stage: str

@dataclass
class PatternValidationReport:
    pattern_id: str
    quality_score: int
    status: str
    feature_analysis: Dict[str, Any]
    threshold_analysis: Dict[str, Any]
    maturity_assessment: Dict[str, Any]
    effectiveness_metrics: Dict[str, Any]
    recommendations: List[str]

class PatternValidator:
    def __init__(self):
        # Thresholds from the prompt
        self.quality_score_min = 70
        self.feature_relevance_min = 0.6
        self.success_rate_min = 0.8
        self.false_rate_max = 0.1

    def analyze_features(self, features: List[float]) -> Dict[str, Any]:
        # Dummy logic: high variance = more noise, more nonzero = more relevant
        if not features:
            return {"relevance_score": 0.0, "feature_importance": [], "noise_level": "high"}
        relevance_score = min(1.0, np.count_nonzero(features) / len(features))
        feature_importance = [f"feature_{i}" for i, v in enumerate(features) if abs(v) > 0.1]
        noise_level = "low" if np.var(features) < 0.2 else "medium" if np.var(features) < 0.5 else "high"
        return {
            "relevance_score": round(relevance_score, 2),
            "feature_importance": feature_importance,
            "noise_level": noise_level
        }

    def analyze_threshold(self, threshold: float, confidence: float) -> Dict[str, Any]:
        # Dummy logic: threshold between 0.5 and 1.0 is good, else needs adjustment
        if 0.5 <= threshold <= 1.0:
            appropriateness = "good"
            suggested_adjustment = None
            reasoning = "Threshold aligns with safety requirements"
        else:
            appropriateness = "review"
            suggested_adjustment = 0.8
            reasoning = "Threshold outside typical safety range"
        return {
            "appropriateness": appropriateness,
            "suggested_adjustment": suggested_adjustment,
            "reasoning": reasoning
        }

    def assess_maturity(self, evolution_stage: str, usage_count: int) -> Dict[str, Any]:
        # Dummy logic: stable if usage_count > 20 and evolution_stage is 'stable'
        if evolution_stage == "stable" and usage_count > 20:
            learning_progress = "complete"
            stability_indicator = "high"
        elif usage_count > 10:
            learning_progress = "in_progress"
            stability_indicator = "medium"
        else:
            learning_progress = "early"
            stability_indicator = "low"
        return {
            "evolution_stage": evolution_stage,
            "learning_progress": learning_progress,
            "stability_indicator": stability_indicator
        }

    def assess_effectiveness(self, success_rate: float) -> Dict[str, Any]:
        # Dummy logic: simulate false positive/negative rates
        false_positive_rate = round((1 - success_rate) * 0.5, 2)
        false_negative_rate = round((1 - success_rate) * 0.3, 2)
        return {
            "operational_success_rate": round(success_rate, 2),
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate
        }

    def compute_quality_score(self, feature_analysis, threshold_analysis, maturity_assessment, effectiveness_metrics) -> int:
        # Weighted sum based on prompt criteria
        score = 0
        if feature_analysis["relevance_score"] > self.feature_relevance_min:
            score += 30
        if threshold_analysis["appropriateness"] == "good":
            score += 20
        if maturity_assessment["stability_indicator"] == "high":
            score += 20
        if effectiveness_metrics["operational_success_rate"] > self.success_rate_min:
            score += 20
        if effectiveness_metrics["false_positive_rate"] < self.false_rate_max and effectiveness_metrics["false_negative_rate"] < self.false_rate_max:
            score += 10
        return score

    def recommend(self, quality_score, feature_analysis, effectiveness_metrics) -> List[str]:
        recs = []
        if quality_score >= 90:
            recs.append("Pattern is well-established and reliable")
        elif quality_score >= 70:
            recs.append("Pattern is valid but could be improved")
        else:
            recs.append("Pattern needs review or retraining")
        if feature_analysis["relevance_score"] < 0.8:
            recs.append("Consider feature engineering for improved sensitivity")
        if effectiveness_metrics["false_positive_rate"] > 0.1:
            recs.append("Reduce false positive rate through threshold tuning")
        return recs

    def validate_pattern(self, pattern: PatternData) -> PatternValidationReport:
        feature_analysis = self.analyze_features(pattern.features)
        threshold_analysis = self.analyze_threshold(pattern.threshold, pattern.confidence)
        maturity_assessment = self.assess_maturity(pattern.evolution_stage, pattern.usage_count)
        effectiveness_metrics = self.assess_effectiveness(pattern.success_rate)
        quality_score = self.compute_quality_score(feature_analysis, threshold_analysis, maturity_assessment, effectiveness_metrics)
        status = "valid" if quality_score >= self.quality_score_min else "needs_review" if quality_score >= 50 else "invalid"
        recommendations = self.recommend(quality_score, feature_analysis, effectiveness_metrics)
        return PatternValidationReport(
            pattern_id=pattern.pattern_id,
            quality_score=quality_score,
            status=status,
            feature_analysis=feature_analysis,
            threshold_analysis=threshold_analysis,
            maturity_assessment=maturity_assessment,
            effectiveness_metrics=effectiveness_metrics,
            recommendations=recommendations
        )

def main():
    parser = argparse.ArgumentParser(description="Validate a learned safety pattern.")
    parser.add_argument('--pattern_id', type=str, required=True)
    parser.add_argument('--features', type=str, required=True, help='Comma-separated list of feature values')
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--confidence', type=float, required=True)
    parser.add_argument('--usage_count', type=int, required=True)
    parser.add_argument('--success_rate', type=float, required=True)
    parser.add_argument('--creation_time', type=float, default=None)
    parser.add_argument('--evolution_stage', type=str, required=True)
    args = parser.parse_args()

    features = [float(x) for x in args.features.split(',')]
    creation_time = args.creation_time if args.creation_time is not None else time.time()
    pattern = PatternData(
        pattern_id=args.pattern_id,
        features=features,
        threshold=args.threshold,
        confidence=args.confidence,
        usage_count=args.usage_count,
        success_rate=args.success_rate,
        creation_time=creation_time,
        evolution_stage=args.evolution_stage
    )
    validator = PatternValidator()
    report = validator.validate_pattern(pattern)
    print(json.dumps(asdict(report), indent=2))

if __name__ == "__main__":
    main() 