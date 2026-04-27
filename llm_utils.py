"""
LLM Utility Module - Three-tier fallback system
1. Gemini (primary)
2. Groq (fallback)
3. Local explanation (final fallback)
"""

import os
from typing import Optional
import re


# ==================== GEMINI ====================
def get_gemini_explanation(bias_findings: dict, api_key: Optional[str] = None) -> Optional[str]:
    """
    Get bias explanation from Google Gemini API (Primary)
    
    Args:
        bias_findings: Dict with bias metrics and analysis results
        api_key: Gemini API key (if None, tries to get from env)
    
    Returns:
        Explanation string or None if failed
    """
    try:
        import google.generativeai as genai
        
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = f"""
        Analyze the following AI bias findings and provide a concise, professional explanation:
        
        Before Mitigation:
        - Group 1 Rate: {bias_findings.get('g1_before', 0):.2%}
        - Group 2 Rate: {bias_findings.get('g2_before', 0):.2%}
        - Disparate Impact Ratio: {bias_findings.get('di_ratio', 0):.2f}
        
        After Mitigation:
        - Group 1 Rate: {bias_findings.get('g1_after', 0):.2%}
        - Group 2 Rate: {bias_findings.get('g2_after', 0):.2%}
        
        Provide a 3-4 sentence explanation of what this means in terms of AI fairness and bias.
        Be professional but accessible to non-technical users.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return None


# ==================== GROQ ====================
def get_groq_explanation(bias_findings: dict, api_key: Optional[str] = None) -> Optional[str]:
    """
    Get bias explanation from Groq API (Fallback)
    
    Args:
        bias_findings: Dict with bias metrics and analysis results
        api_key: Groq API key (if None, tries to get from env)
    
    Returns:
        Explanation string or None if failed
    """
    try:
        from groq import Groq
        
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        
        client = Groq(api_key=api_key)
        
        prompt = f"""
        Analyze the following AI bias findings and provide a concise, professional explanation:
        
        Before Mitigation:
        - Group 1 Rate: {bias_findings.get('g1_before', 0):.2%}
        - Group 2 Rate: {bias_findings.get('g2_before', 0):.2%}
        - Disparate Impact Ratio: {bias_findings.get('di_ratio', 0):.2f}
        
        After Mitigation:
        - Group 1 Rate: {bias_findings.get('g1_after', 0):.2%}
        - Group 2 Rate: {bias_findings.get('g2_after', 0):.2%}
        
        Provide a 3-4 sentence explanation of what this means in terms of AI fairness and bias.
        Be professional but accessible to non-technical users.
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Groq API error: {str(e)}")
        return None


# ==================== LOCAL EXPLANATION ====================
def get_local_explanation(bias_findings: dict) -> str:
    """
    Generate local explanation without API calls (Final Fallback)
    
    Args:
        bias_findings: Dict with bias metrics and analysis results
    
    Returns:
        Explanation string
    """
    g1_before = bias_findings.get('g1_before', 0)
    g2_before = bias_findings.get('g2_before', 0)
    g1_after = bias_findings.get('g1_after', 0)
    g2_after = bias_findings.get('g2_after', 0)
    di_ratio = bias_findings.get('di_ratio', 0)
    
    # Analyze before mitigation
    if di_ratio < 0.8:
        bias_status = "exhibits significant bias"
        severity = "concerning" if di_ratio < 0.6 else "notable"
    elif di_ratio <= 1.25:
        bias_status = "appears reasonably fair"
        severity = "acceptable"
    else:
        bias_status = "may favor one group"
        severity = "potential"
    
    # Calculate improvement
    diff_before = abs(g1_before - g2_before)
    diff_after = abs(g1_after - g2_after)
    improvement = ((diff_before - diff_after) / diff_before * 100) if diff_before > 0 else 0
    
    explanation = f"""
    The model {bias_status}, with a disparate impact ratio of {di_ratio:.2f}. 
    Before mitigation, Group 1 had a {g1_before:.1%} positive rate while Group 2 had {g2_before:.1%}, 
    indicating a {severity} fairness concern. After removing the sensitive attribute, 
    the difference reduced to {abs(g1_after - g2_after):.1%} (approximately {improvement:.0f}% improvement), 
    suggesting that the sensitive attribute significantly influenced model predictions.
    """
    
    return explanation.strip()


# ==================== MAIN FALLBACK SYSTEM ====================
def get_bias_explanation(
    bias_findings: dict,
    gemini_key: Optional[str] = None,
    groq_key: Optional[str] = None,
    use_fallback: bool = True
) -> dict:
    """
    Three-tier LLM fallback system for bias explanation
    1. Try Gemini (primary)
    2. Try Groq (fallback)
    3. Use local explanation (final fallback)
    
    Args:
        bias_findings: Dict with bias metrics (g1_before, g2_before, g1_after, g2_after, di_ratio)
        gemini_key: Gemini API key
        groq_key: Groq API key
        use_fallback: Whether to use local explanation if APIs fail
    
    Returns:
        Dict with explanation and source info:
        {
            'explanation': str,
            'source': 'gemini' | 'groq' | 'local',
            'success': bool
        }
    """
    
    # Try Gemini first
    explanation = get_gemini_explanation(bias_findings, gemini_key)
    if explanation:
        return {
            'explanation': explanation,
            'source': 'gemini',
            'success': True
        }
    
    # Fallback to Groq
    explanation = get_groq_explanation(bias_findings, groq_key)
    if explanation:
        return {
            'explanation': explanation,
            'source': 'groq',
            'success': True
        }
    
    # Final fallback to local
    if use_fallback:
        explanation = get_local_explanation(bias_findings)
        return {
            'explanation': explanation,
            'source': 'local',
            'success': True
        }
    
    return {
        'explanation': None,
        'source': None,
        'success': False
    }
