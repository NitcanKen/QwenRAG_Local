"""
Analytics service for customer churn analysis.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import json
import hashlib
from functools import lru_cache

from app.core.database import DatabaseManager, supabase, AsyncSessionLocal
from app.core.logging import get_logger
from app.core.exceptions import DatabaseConnectionError
from app.core.config import settings

logger = get_logger(__name__)

# Simple in-memory cache for analytics results (fallback when Redis not available)
_cache = {}

def get_cache_key(function_name: str, **kwargs) -> str:
    """
    Generate cache key for function call.
    
    Args:
        function_name: Name of the analytics function
        **kwargs: Additional parameters for cache key
        
    Returns:
        SHA256 hash of the cache key
    """
    key_data = {
        "function": function_name,
        "params": kwargs,
        "timestamp": datetime.now().strftime("%Y-%m-%d-%H")  # Cache per hour
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()

async def get_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Get cached result.
    
    Args:
        cache_key: Cache key to look up
        
    Returns:
        Cached result or None
    """
    try:
        # Try Redis first (if available)
        try:
            import redis
            redis_client = redis.Redis.from_url(settings.REDIS_URL)
            cached_data = redis_client.get(cache_key)
            if cached_data:
                logger.info(f"Cache hit (Redis): {cache_key[:8]}...")
                return json.loads(cached_data)
        except Exception as e:
            logger.debug(f"Redis cache not available: {e}")
        
        # Fallback to in-memory cache
        if cache_key in _cache:
            cached_entry = _cache[cache_key]
            # Check if cache entry is still valid (1 hour)
            if datetime.now() - cached_entry["timestamp"] < timedelta(hours=1):
                logger.info(f"Cache hit (memory): {cache_key[:8]}...")
                return cached_entry["data"]
            else:
                # Remove expired entry
                del _cache[cache_key]
        
        return None
        
    except Exception as e:
        logger.warning(f"Error accessing cache: {e}")
        return None

async def set_cached_result(cache_key: str, data: Dict[str, Any], ttl_seconds: int = 3600) -> None:
    """
    Set cached result.
    
    Args:
        cache_key: Cache key
        data: Data to cache
        ttl_seconds: Time to live in seconds (default 1 hour)
    """
    try:
        # Try Redis first (if available)
        try:
            import redis
            redis_client = redis.Redis.from_url(settings.REDIS_URL)
            redis_client.setex(cache_key, ttl_seconds, json.dumps(data))
            logger.info(f"Cache set (Redis): {cache_key[:8]}...")
            return
        except Exception as e:
            logger.debug(f"Redis cache not available: {e}")
        
        # Fallback to in-memory cache
        _cache[cache_key] = {
            "data": data,
            "timestamp": datetime.now()
        }
        logger.info(f"Cache set (memory): {cache_key[:8]}...")
        
        # Clean up old entries (keep only last 50)
        if len(_cache) > 50:
            oldest_keys = sorted(_cache.keys(), key=lambda k: _cache[k]["timestamp"])[:10]
            for key in oldest_keys:
                del _cache[key]
        
    except Exception as e:
        logger.warning(f"Error setting cache: {e}")

def cache_analytics_result(function_name: str, ttl_seconds: int = 3600):
    """
    Decorator to cache analytics function results.
    
    Args:
        function_name: Name of the function for cache key
        ttl_seconds: Time to live in seconds
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = get_cache_key(function_name, **kwargs)
            
            # Try to get cached result
            cached_result = await get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache the result
            await set_cached_result(cache_key, result, ttl_seconds)
            
            return result
        return wrapper
    return decorator


class ChurnAnalyticsService:
    """Service class for customer churn analytics calculations."""
    
    @staticmethod
    @cache_analytics_result("churn_rate_by_tenure", ttl_seconds=1800)  # 30 minutes cache
    async def get_churn_rate_by_tenure() -> Dict[str, Any]:
        """
        Calculate churn rates by tenure groups.
        
        Returns:
            Dictionary containing churn rates by tenure
        """
        try:
            logger.info("Calculating churn rate by tenure")
            
            if supabase:
                # Supabase query for tenure analysis
                response = supabase.table("customers").select(
                    "tenure_group, churn"
                ).execute()
                
                if not response.data:
                    logger.warning("No customer data found")
                    return {"error": "No data available"}
                
                # Group and calculate churn rates
                tenure_groups = {}
                for row in response.data:
                    group = row.get("tenure_group", "Unknown")
                    churn = row.get("churn", 0)
                    
                    if group not in tenure_groups:
                        tenure_groups[group] = {"total": 0, "churned": 0}
                    
                    tenure_groups[group]["total"] += 1
                    if churn == 1:
                        tenure_groups[group]["churned"] += 1
                
                # Calculate rates
                result = {}
                for group, data in tenure_groups.items():
                    rate = data["churned"] / data["total"] if data["total"] > 0 else 0
                    result[group] = {
                        "churn_rate": round(rate, 3),
                        "total_customers": data["total"],
                        "churned_customers": data["churned"],
                        "retained_customers": data["total"] - data["churned"]
                    }
                
                return {
                    "tenure_analysis": result,
                    "total_customers": sum(data["total"] for data in tenure_groups.values()),
                    "overall_churn_rate": round(
                        sum(data["churned"] for data in tenure_groups.values()) /
                        sum(data["total"] for data in tenure_groups.values()), 3
                    ) if tenure_groups else 0
                }
                
            elif AsyncSessionLocal:
                # PostgreSQL query for tenure analysis
                async with AsyncSessionLocal() as session:
                    from sqlalchemy import text
                    
                    query = text("""
                        SELECT 
                            tenure_group,
                            COUNT(*) as total_customers,
                            SUM(churn) as churned_customers,
                            ROUND(AVG(churn::float), 3) as churn_rate
                        FROM customers 
                        WHERE tenure_group IS NOT NULL
                        GROUP BY tenure_group
                        ORDER BY tenure_group
                    """)
                    
                    result = await session.execute(query)
                    rows = result.fetchall()
                    
                    if not rows:
                        logger.warning("No customer data found")
                        return {"error": "No data available"}
                    
                    tenure_analysis = {}
                    total_customers = 0
                    total_churned = 0
                    
                    for row in rows:
                        group = row.tenure_group
                        total = row.total_customers
                        churned = row.churned_customers or 0
                        
                        tenure_analysis[group] = {
                            "churn_rate": row.churn_rate or 0,
                            "total_customers": total,
                            "churned_customers": churned,
                            "retained_customers": total - churned
                        }
                        
                        total_customers += total
                        total_churned += churned
                    
                    return {
                        "tenure_analysis": tenure_analysis,
                        "total_customers": total_customers,
                        "overall_churn_rate": round(total_churned / total_customers, 3) if total_customers > 0 else 0
                    }
            else:
                # Return mock data when no database connection
                logger.warning("No database connection - returning mock data")
                return {
                    "tenure_analysis": {
                        "0-1 years": {
                            "churn_rate": 0.356,
                            "total_customers": 2456,
                            "churned_customers": 874,
                            "retained_customers": 1582
                        },
                        "1-2 years": {
                            "churn_rate": 0.289,
                            "total_customers": 1847,
                            "churned_customers": 534,
                            "retained_customers": 1313
                        },
                        "2-3 years": {
                            "churn_rate": 0.198,
                            "total_customers": 1523,
                            "churned_customers": 301,
                            "retained_customers": 1222
                        },
                        "3+ years": {
                            "churn_rate": 0.145,
                            "total_customers": 1217,
                            "churned_customers": 176,
                            "retained_customers": 1041
                        }
                    },
                    "total_customers": 7043,
                    "overall_churn_rate": 0.267,
                    "note": "Mock data - no database connection"
                }
                
        except Exception as e:
            logger.error(f"Error calculating churn rate by tenure: {e}")
            raise DatabaseConnectionError(f"Failed to calculate tenure analysis: {str(e)}")
    
    @staticmethod
    @cache_analytics_result("churn_rate_by_contract", ttl_seconds=1800)  # 30 minutes cache
    async def get_churn_rate_by_contract() -> Dict[str, Any]:
        """
        Calculate churn rates by contract type.
        
        Returns:
            Dictionary containing churn rates by contract
        """
        try:
            logger.info("Calculating churn rate by contract")
            
            if supabase:
                # Supabase query for contract analysis
                response = supabase.table("customers").select(
                    "contract, churn"
                ).execute()
                
                if not response.data:
                    return {"error": "No data available"}
                
                # Group and calculate churn rates
                contract_groups = {}
                for row in response.data:
                    contract = row.get("contract", "Unknown")
                    churn = row.get("churn", 0)
                    
                    if contract not in contract_groups:
                        contract_groups[contract] = {"total": 0, "churned": 0}
                    
                    contract_groups[contract]["total"] += 1
                    if churn == 1:
                        contract_groups[contract]["churned"] += 1
                
                # Calculate rates
                result = {}
                for contract, data in contract_groups.items():
                    rate = data["churned"] / data["total"] if data["total"] > 0 else 0
                    result[contract] = {
                        "churn_rate": round(rate, 3),
                        "total_customers": data["total"],
                        "churned_customers": data["churned"],
                        "retained_customers": data["total"] - data["churned"]
                    }
                
                return {"contract_analysis": result}
                
            elif AsyncSessionLocal:
                # PostgreSQL query for contract analysis
                async with AsyncSessionLocal() as session:
                    from sqlalchemy import text
                    
                    query = text("""
                        SELECT 
                            contract,
                            COUNT(*) as total_customers,
                            SUM(churn) as churned_customers,
                            ROUND(AVG(churn::float), 3) as churn_rate
                        FROM customers 
                        WHERE contract IS NOT NULL
                        GROUP BY contract
                        ORDER BY churn_rate DESC
                    """)
                    
                    result = await session.execute(query)
                    rows = result.fetchall()
                    
                    contract_analysis = {}
                    for row in rows:
                        contract = row.contract
                        total = row.total_customers
                        churned = row.churned_customers or 0
                        
                        contract_analysis[contract] = {
                            "churn_rate": row.churn_rate or 0,
                            "total_customers": total,
                            "churned_customers": churned,
                            "retained_customers": total - churned
                        }
                    
                    return {"contract_analysis": contract_analysis}
            else:
                # Return mock data
                logger.warning("No database connection - returning mock data")
                return {
                    "contract_analysis": {
                        "Month-to-month": {
                            "churn_rate": 0.427,
                            "total_customers": 3875,
                            "churned_customers": 1655,
                            "retained_customers": 2220
                        },
                        "One year": {
                            "churn_rate": 0.113,
                            "total_customers": 1473,
                            "churned_customers": 166,
                            "retained_customers": 1307
                        },
                        "Two year": {
                            "churn_rate": 0.028,
                            "total_customers": 1695,
                            "churned_customers": 48,
                            "retained_customers": 1647
                        }
                    },
                    "note": "Mock data - no database connection"
                }
                
        except Exception as e:
            logger.error(f"Error calculating churn rate by contract: {e}")
            raise DatabaseConnectionError(f"Failed to calculate contract analysis: {str(e)}")
    
    @staticmethod
    @cache_analytics_result("churn_rate_by_payment_method", ttl_seconds=1800)  # 30 minutes cache
    async def get_churn_rate_by_payment_method() -> Dict[str, Any]:
        """
        Calculate churn rates by payment method.
        
        Returns:
            Dictionary containing churn rates by payment method
        """
        try:
            logger.info("Calculating churn rate by payment method")
            
            if supabase:
                # Supabase query for payment method analysis
                response = supabase.table("customers").select(
                    "payment_method, churn"
                ).execute()
                
                if not response.data:
                    return {"error": "No data available"}
                
                # Group and calculate churn rates
                payment_groups = {}
                for row in response.data:
                    payment = row.get("payment_method", "Unknown")
                    churn = row.get("churn", 0)
                    
                    if payment not in payment_groups:
                        payment_groups[payment] = {"total": 0, "churned": 0}
                    
                    payment_groups[payment]["total"] += 1
                    if churn == 1:
                        payment_groups[payment]["churned"] += 1
                
                # Calculate rates
                result = {}
                for payment, data in payment_groups.items():
                    rate = data["churned"] / data["total"] if data["total"] > 0 else 0
                    result[payment] = {
                        "churn_rate": round(rate, 3),
                        "total_customers": data["total"],
                        "churned_customers": data["churned"],
                        "retained_customers": data["total"] - data["churned"]
                    }
                
                return {"payment_method_analysis": result}
                
            elif AsyncSessionLocal:
                # PostgreSQL query for payment method analysis
                async with AsyncSessionLocal() as session:
                    from sqlalchemy import text
                    
                    query = text("""
                        SELECT 
                            payment_method,
                            COUNT(*) as total_customers,
                            SUM(churn) as churned_customers,
                            ROUND(AVG(churn::float), 3) as churn_rate
                        FROM customers 
                        WHERE payment_method IS NOT NULL
                        GROUP BY payment_method
                        ORDER BY churn_rate DESC
                    """)
                    
                    result = await session.execute(query)
                    rows = result.fetchall()
                    
                    payment_analysis = {}
                    for row in rows:
                        payment = row.payment_method
                        total = row.total_customers
                        churned = row.churned_customers or 0
                        
                        payment_analysis[payment] = {
                            "churn_rate": row.churn_rate or 0,
                            "total_customers": total,
                            "churned_customers": churned,
                            "retained_customers": total - churned
                        }
                    
                    return {"payment_method_analysis": payment_analysis}
            else:
                # Return mock data
                logger.warning("No database connection - returning mock data")
                return {
                    "payment_method_analysis": {
                        "Electronic check": {
                            "churn_rate": 0.453,
                            "total_customers": 2365,
                            "churned_customers": 1071,
                            "retained_customers": 1294
                        },
                        "Mailed check": {
                            "churn_rate": 0.191,
                            "total_customers": 1612,
                            "churned_customers": 308,
                            "retained_customers": 1304
                        },
                        "Bank transfer (automatic)": {
                            "churn_rate": 0.167,
                            "total_customers": 1544,
                            "churned_customers": 258,
                            "retained_customers": 1286
                        },
                        "Credit card (automatic)": {
                            "churn_rate": 0.152,
                            "total_customers": 1522,
                            "churned_customers": 232,
                            "retained_customers": 1290
                        }
                    },
                    "note": "Mock data - no database connection"
                }
                
        except Exception as e:
            logger.error(f"Error calculating churn rate by payment method: {e}")
            raise DatabaseConnectionError(f"Failed to calculate payment method analysis: {str(e)}")
    
    @staticmethod
    @cache_analytics_result("overall_churn_metrics", ttl_seconds=1800)  # 30 minutes cache
    async def get_overall_churn_metrics() -> Dict[str, Any]:
        """
        Get overall churn metrics and KPIs.
        
        Returns:
            Dictionary containing overall churn statistics
        """
        try:
            logger.info("Calculating overall churn metrics")
            
            if supabase:
                # Supabase query for overall metrics
                response = supabase.table("customers").select(
                    "churn, monthly_charges, total_charges"
                ).execute()
                
                if not response.data:
                    return {"error": "No data available"}
                
                total_customers = len(response.data)
                churned_customers = sum(1 for row in response.data if row.get("churn") == 1)
                retained_customers = total_customers - churned_customers
                
                # Calculate revenue metrics
                churned_revenue = sum(
                    row.get("monthly_charges", 0) 
                    for row in response.data 
                    if row.get("churn") == 1
                )
                
                total_revenue = sum(
                    row.get("monthly_charges", 0) 
                    for row in response.data
                )
                
                return {
                    "total_customers": total_customers,
                    "churned_customers": churned_customers,
                    "retained_customers": retained_customers,
                    "churn_rate": round(churned_customers / total_customers, 3) if total_customers > 0 else 0,
                    "retention_rate": round(retained_customers / total_customers, 3) if total_customers > 0 else 0,
                    "monthly_revenue_lost": round(churned_revenue, 2),
                    "total_monthly_revenue": round(total_revenue, 2),
                    "revenue_impact_percentage": round(churned_revenue / total_revenue * 100, 2) if total_revenue > 0 else 0
                }
                
            elif AsyncSessionLocal:
                # PostgreSQL query for overall metrics
                async with AsyncSessionLocal() as session:
                    from sqlalchemy import text
                    
                    query = text("""
                        SELECT 
                            COUNT(*) as total_customers,
                            SUM(churn) as churned_customers,
                            SUM(CASE WHEN churn = 0 THEN 1 ELSE 0 END) as retained_customers,
                            ROUND(AVG(churn::float), 3) as churn_rate,
                            SUM(CASE WHEN churn = 1 THEN monthly_charges ELSE 0 END) as churned_revenue,
                            SUM(monthly_charges) as total_revenue
                        FROM customers
                    """)
                    
                    result = await session.execute(query)
                    row = result.fetchone()
                    
                    if not row:
                        return {"error": "No data available"}
                    
                    total_customers = row.total_customers or 0
                    churned_customers = row.churned_customers or 0
                    retained_customers = row.retained_customers or 0
                    churn_rate = row.churn_rate or 0
                    churned_revenue = float(row.churned_revenue or 0)
                    total_revenue = float(row.total_revenue or 0)
                    
                    return {
                        "total_customers": total_customers,
                        "churned_customers": churned_customers,
                        "retained_customers": retained_customers,
                        "churn_rate": churn_rate,
                        "retention_rate": round(1 - churn_rate, 3),
                        "monthly_revenue_lost": round(churned_revenue, 2),
                        "total_monthly_revenue": round(total_revenue, 2),
                        "revenue_impact_percentage": round(churned_revenue / total_revenue * 100, 2) if total_revenue > 0 else 0
                    }
            else:
                # Return mock data
                logger.warning("No database connection - returning mock data")
                return {
                    "total_customers": 7043,
                    "churned_customers": 1869,
                    "retained_customers": 5174,
                    "churn_rate": 0.265,
                    "retention_rate": 0.735,
                    "monthly_revenue_lost": 139169.36,
                    "total_monthly_revenue": 456201.85,
                    "revenue_impact_percentage": 30.51,
                    "note": "Mock data - no database connection"
                }
                
        except Exception as e:
            logger.error(f"Error calculating overall churn metrics: {e}")
            raise DatabaseConnectionError(f"Failed to calculate overall metrics: {str(e)}")
    
    @staticmethod
    @cache_analytics_result("demographic_analysis", ttl_seconds=1800)  # 30 minutes cache
    async def get_demographic_analysis() -> Dict[str, Any]:
        """
        Get churn analysis by demographic factors.
        
        Returns:
            Dictionary containing demographic churn analysis
        """
        try:
            logger.info("Calculating demographic churn analysis")
            
            if supabase:
                # Get gender analysis
                gender_response = supabase.table("customers").select("gender, churn").execute()
                
                # Get senior citizen analysis
                senior_response = supabase.table("customers").select("senior_citizen, churn").execute()
                
                # Get partner analysis
                partner_response = supabase.table("customers").select("partner, churn").execute()
                
                # Get dependents analysis
                dependents_response = supabase.table("customers").select("dependents, churn").execute()
                
                def analyze_dimension(data, dimension_field):
                    """Helper function to analyze a demographic dimension."""
                    groups = {}
                    for row in data:
                        key = row.get(dimension_field, "Unknown")
                        churn = row.get("churn", 0)
                        
                        if key not in groups:
                            groups[key] = {"total": 0, "churned": 0}
                        
                        groups[key]["total"] += 1
                        if churn == 1:
                            groups[key]["churned"] += 1
                    
                    result = {}
                    for key, data in groups.items():
                        rate = data["churned"] / data["total"] if data["total"] > 0 else 0
                        result[key] = {
                            "churn_rate": round(rate, 3),
                            "total_customers": data["total"],
                            "churned_customers": data["churned"],
                            "retained_customers": data["total"] - data["churned"]
                        }
                    return result
                
                return {
                    "gender_analysis": analyze_dimension(gender_response.data, "gender"),
                    "senior_citizen_analysis": analyze_dimension(senior_response.data, "senior_citizen"),
                    "partner_analysis": analyze_dimension(partner_response.data, "partner"),
                    "dependents_analysis": analyze_dimension(dependents_response.data, "dependents")
                }
                
            elif AsyncSessionLocal:
                # PostgreSQL queries for demographic analysis
                async with AsyncSessionLocal() as session:
                    from sqlalchemy import text
                    
                    # Gender analysis
                    gender_query = text("""
                        SELECT 
                            gender,
                            COUNT(*) as total_customers,
                            SUM(churn) as churned_customers,
                            ROUND(AVG(churn::float), 3) as churn_rate
                        FROM customers 
                        WHERE gender IS NOT NULL
                        GROUP BY gender
                    """)
                    
                    # Senior citizen analysis
                    senior_query = text("""
                        SELECT 
                            CASE WHEN senior_citizen = 1 THEN 'Senior' ELSE 'Non-Senior' END as senior_status,
                            COUNT(*) as total_customers,
                            SUM(churn) as churned_customers,
                            ROUND(AVG(churn::float), 3) as churn_rate
                        FROM customers 
                        WHERE senior_citizen IS NOT NULL
                        GROUP BY senior_citizen
                    """)
                    
                    # Partner analysis
                    partner_query = text("""
                        SELECT 
                            partner,
                            COUNT(*) as total_customers,
                            SUM(churn) as churned_customers,
                            ROUND(AVG(churn::float), 3) as churn_rate
                        FROM customers 
                        WHERE partner IS NOT NULL
                        GROUP BY partner
                    """)
                    
                    # Dependents analysis
                    dependents_query = text("""
                        SELECT 
                            dependents,
                            COUNT(*) as total_customers,
                            SUM(churn) as churned_customers,
                            ROUND(AVG(churn::float), 3) as churn_rate
                        FROM customers 
                        WHERE dependents IS NOT NULL
                        GROUP BY dependents
                    """)
                    
                    def process_query_result(result):
                        """Helper to process query results."""
                        analysis = {}
                        for row in result.fetchall():
                            key = row[0]
                            total = row.total_customers
                            churned = row.churned_customers or 0
                            
                            analysis[key] = {
                                "churn_rate": row.churn_rate or 0,
                                "total_customers": total,
                                "churned_customers": churned,
                                "retained_customers": total - churned
                            }
                        return analysis
                    
                    # Execute all queries
                    gender_result = await session.execute(gender_query)
                    senior_result = await session.execute(senior_query)
                    partner_result = await session.execute(partner_query)
                    dependents_result = await session.execute(dependents_query)
                    
                    return {
                        "gender_analysis": process_query_result(gender_result),
                        "senior_citizen_analysis": process_query_result(senior_result),
                        "partner_analysis": process_query_result(partner_result),
                        "dependents_analysis": process_query_result(dependents_result)
                    }
            else:
                # Return mock data
                logger.warning("No database connection - returning mock demographic data")
                return {
                    "gender_analysis": {
                        "Female": {
                            "churn_rate": 0.270,
                            "total_customers": 3488,
                            "churned_customers": 939,
                            "retained_customers": 2549
                        },
                        "Male": {
                            "churn_rate": 0.263,
                            "total_customers": 3555,
                            "churned_customers": 930,
                            "retained_customers": 2625
                        }
                    },
                    "senior_citizen_analysis": {
                        "Senior": {
                            "churn_rate": 0.417,
                            "total_customers": 1142,
                            "churned_customers": 476,
                            "retained_customers": 666
                        },
                        "Non-Senior": {
                            "churn_rate": 0.238,
                            "total_customers": 5901,
                            "churned_customers": 1393,
                            "retained_customers": 4508
                        }
                    },
                    "partner_analysis": {
                        "Yes": {
                            "churn_rate": 0.197,
                            "total_customers": 3402,
                            "churned_customers": 669,
                            "retained_customers": 2733
                        },
                        "No": {
                            "churn_rate": 0.330,
                            "total_customers": 3641,
                            "churned_customers": 1200,
                            "retained_customers": 2441
                        }
                    },
                    "dependents_analysis": {
                        "Yes": {
                            "churn_rate": 0.156,
                            "total_customers": 2110,
                            "churned_customers": 329,
                            "retained_customers": 1781
                        },
                        "No": {
                            "churn_rate": 0.312,
                            "total_customers": 4933,
                            "churned_customers": 1540,
                            "retained_customers": 3393
                        }
                    },
                    "note": "Mock data - no database connection"
                }
                
        except Exception as e:
            logger.error(f"Error calculating demographic analysis: {e}")
            raise DatabaseConnectionError(f"Failed to calculate demographic analysis: {str(e)}")
    
    @staticmethod
    @cache_analytics_result("service_impact_analysis", ttl_seconds=1800)  # 30 minutes cache
    async def get_service_impact_analysis() -> Dict[str, Any]:
        """
        Get churn analysis by service factors.
        
        Returns:
            Dictionary containing service impact churn analysis
        """
        try:
            logger.info("Calculating service impact churn analysis")
            
            if supabase:
                # Get internet service analysis
                internet_response = supabase.table("customers").select("internet_service, churn").execute()
                
                # Get streaming services analysis
                streaming_tv_response = supabase.table("customers").select("streaming_tv, churn").execute()
                streaming_movies_response = supabase.table("customers").select("streaming_movies, churn").execute()
                
                # Get security services analysis
                online_security_response = supabase.table("customers").select("online_security, churn").execute()
                online_backup_response = supabase.table("customers").select("online_backup, churn").execute()
                device_protection_response = supabase.table("customers").select("device_protection, churn").execute()
                tech_support_response = supabase.table("customers").select("tech_support, churn").execute()
                
                def analyze_service_dimension(data, dimension_field):
                    """Helper function to analyze a service dimension."""
                    groups = {}
                    for row in data:
                        key = row.get(dimension_field, "Unknown")
                        churn = row.get("churn", 0)
                        
                        if key not in groups:
                            groups[key] = {"total": 0, "churned": 0}
                        
                        groups[key]["total"] += 1
                        if churn == 1:
                            groups[key]["churned"] += 1
                    
                    result = {}
                    for key, data in groups.items():
                        rate = data["churned"] / data["total"] if data["total"] > 0 else 0
                        result[key] = {
                            "churn_rate": round(rate, 3),
                            "total_customers": data["total"],
                            "churned_customers": data["churned"],
                            "retained_customers": data["total"] - data["churned"]
                        }
                    return result
                
                return {
                    "internet_service_analysis": analyze_service_dimension(internet_response.data, "internet_service"),
                    "streaming_tv_analysis": analyze_service_dimension(streaming_tv_response.data, "streaming_tv"),
                    "streaming_movies_analysis": analyze_service_dimension(streaming_movies_response.data, "streaming_movies"),
                    "online_security_analysis": analyze_service_dimension(online_security_response.data, "online_security"),
                    "online_backup_analysis": analyze_service_dimension(online_backup_response.data, "online_backup"),
                    "device_protection_analysis": analyze_service_dimension(device_protection_response.data, "device_protection"),
                    "tech_support_analysis": analyze_service_dimension(tech_support_response.data, "tech_support")
                }
                
            elif AsyncSessionLocal:
                # PostgreSQL queries for service analysis
                async with AsyncSessionLocal() as session:
                    from sqlalchemy import text
                    
                    services = [
                        "internet_service", "streaming_tv", "streaming_movies",
                        "online_security", "online_backup", "device_protection", "tech_support"
                    ]
                    
                    def process_service_query_result(result):
                        """Helper to process service query results."""
                        analysis = {}
                        for row in result.fetchall():
                            key = row[0]
                            total = row.total_customers
                            churned = row.churned_customers or 0
                            
                            analysis[key] = {
                                "churn_rate": row.churn_rate or 0,
                                "total_customers": total,
                                "churned_customers": churned,
                                "retained_customers": total - churned
                            }
                        return analysis
                    
                    results = {}
                    for service in services:
                        query = text(f"""
                            SELECT 
                                {service},
                                COUNT(*) as total_customers,
                                SUM(churn) as churned_customers,
                                ROUND(AVG(churn::float), 3) as churn_rate
                            FROM customers 
                            WHERE {service} IS NOT NULL
                            GROUP BY {service}
                            ORDER BY churn_rate DESC
                        """)
                        
                        result = await session.execute(query)
                        results[f"{service}_analysis"] = process_service_query_result(result)
                    
                    return results
            else:
                # Return mock data
                logger.warning("No database connection - returning mock service data")
                return {
                    "internet_service_analysis": {
                        "Fiber optic": {
                            "churn_rate": 0.419,
                            "total_customers": 3096,
                            "churned_customers": 1297,
                            "retained_customers": 1799
                        },
                        "DSL": {
                            "churn_rate": 0.191,
                            "total_customers": 2421,
                            "churned_customers": 459,
                            "retained_customers": 1962
                        },
                        "No": {
                            "churn_rate": 0.074,
                            "total_customers": 1526,
                            "churned_customers": 113,
                            "retained_customers": 1413
                        }
                    },
                    "streaming_tv_analysis": {
                        "No": {
                            "churn_rate": 0.330,
                            "total_customers": 2810,
                            "churned_customers": 927,
                            "retained_customers": 1883
                        },
                        "Yes": {
                            "churn_rate": 0.201,
                            "total_customers": 2707,
                            "churned_customers": 544,
                            "retained_customers": 2163
                        },
                        "No internet service": {
                            "churn_rate": 0.074,
                            "total_customers": 1526,
                            "churned_customers": 113,
                            "retained_customers": 1413
                        }
                    },
                    "streaming_movies_analysis": {
                        "No": {
                            "churn_rate": 0.336,
                            "total_customers": 2785,
                            "churned_customers": 935,
                            "retained_customers": 1850
                        },
                        "Yes": {
                            "churn_rate": 0.200,
                            "total_customers": 2732,
                            "churned_customers": 546,
                            "retained_customers": 2186
                        },
                        "No internet service": {
                            "churn_rate": 0.074,
                            "total_customers": 1526,
                            "churned_customers": 113,
                            "retained_customers": 1413
                        }
                    },
                    "online_security_analysis": {
                        "No": {
                            "churn_rate": 0.417,
                            "total_customers": 3498,
                            "churned_customers": 1458,
                            "retained_customers": 2040
                        },
                        "Yes": {
                            "churn_rate": 0.147,
                            "total_customers": 2019,
                            "churned_customers": 297,
                            "retained_customers": 1722
                        },
                        "No internet service": {
                            "churn_rate": 0.074,
                            "total_customers": 1526,
                            "churned_customers": 113,
                            "retained_customers": 1413
                        }
                    },
                    "online_backup_analysis": {
                        "No": {
                            "churn_rate": 0.398,
                            "total_customers": 3088,
                            "churned_customers": 1229,
                            "retained_customers": 1859
                        },
                        "Yes": {
                            "churn_rate": 0.218,
                            "total_customers": 2429,
                            "churned_customers": 529,
                            "retained_customers": 1900
                        },
                        "No internet service": {
                            "churn_rate": 0.074,
                            "total_customers": 1526,
                            "churned_customers": 113,
                            "retained_customers": 1413
                        }
                    },
                    "device_protection_analysis": {
                        "No": {
                            "churn_rate": 0.391,
                            "total_customers": 3095,
                            "churned_customers": 1211,
                            "retained_customers": 1884
                        },
                        "Yes": {
                            "churn_rate": 0.229,
                            "total_customers": 2422,
                            "churned_customers": 555,
                            "retained_customers": 1867
                        },
                        "No internet service": {
                            "churn_rate": 0.074,
                            "total_customers": 1526,
                            "churned_customers": 113,
                            "retained_customers": 1413
                        }
                    },
                    "tech_support_analysis": {
                        "No": {
                            "churn_rate": 0.417,
                            "total_customers": 3473,
                            "churned_customers": 1446,
                            "retained_customers": 2027
                        },
                        "Yes": {
                            "churn_rate": 0.154,
                            "total_customers": 2044,
                            "churned_customers": 315,
                            "retained_customers": 1729
                        },
                        "No internet service": {
                            "churn_rate": 0.074,
                            "total_customers": 1526,
                            "churned_customers": 113,
                            "retained_customers": 1413
                        }
                    },
                    "note": "Mock data - no database connection"
                }
                
        except Exception as e:
            logger.error(f"Error calculating service impact analysis: {e}")
            raise DatabaseConnectionError(f"Failed to calculate service impact analysis: {str(e)}")
    
    @staticmethod
    @cache_analytics_result("financial_metrics", ttl_seconds=1800)  # 30 minutes cache
    async def get_financial_metrics() -> Dict[str, Any]:
        """
        Get financial metrics related to churn.
        
        Returns:
            Dictionary containing financial analysis
        """
        try:
            logger.info("Calculating financial metrics")
            
            if supabase:
                # Get all customer financial data
                response = supabase.table("customers").select(
                    "churn, monthly_charges, total_charges"
                ).execute()
                
                if not response.data:
                    return {"error": "No data available"}
                
                # Calculate financial metrics
                churned_customers = [row for row in response.data if row.get("churn") == 1]
                retained_customers = [row for row in response.data if row.get("churn") == 0]
                
                # Average charges analysis
                churned_monthly_avg = sum(row.get("monthly_charges", 0) for row in churned_customers) / len(churned_customers) if churned_customers else 0
                retained_monthly_avg = sum(row.get("monthly_charges", 0) for row in retained_customers) / len(retained_customers) if retained_customers else 0
                
                churned_total_avg = sum(row.get("total_charges", 0) for row in churned_customers) / len(churned_customers) if churned_customers else 0
                retained_total_avg = sum(row.get("total_charges", 0) for row in retained_customers) / len(retained_customers) if retained_customers else 0
                
                # Revenue impact
                monthly_revenue_lost = sum(row.get("monthly_charges", 0) for row in churned_customers)
                total_revenue_lost = sum(row.get("total_charges", 0) for row in churned_customers)
                total_monthly_revenue = sum(row.get("monthly_charges", 0) for row in response.data)
                
                # Charges distribution
                charges_ranges = {"0-30": 0, "30-60": 0, "60-90": 0, "90+": 0}
                for row in response.data:
                    charge = row.get("monthly_charges", 0)
                    if charge < 30:
                        charges_ranges["0-30"] += 1
                    elif charge < 60:
                        charges_ranges["30-60"] += 1
                    elif charge < 90:
                        charges_ranges["60-90"] += 1
                    else:
                        charges_ranges["90+"] += 1
                
                return {
                    "average_charges": {
                        "churned_monthly_avg": round(churned_monthly_avg, 2),
                        "retained_monthly_avg": round(retained_monthly_avg, 2),
                        "churned_total_avg": round(churned_total_avg, 2),
                        "retained_total_avg": round(retained_total_avg, 2)
                    },
                    "revenue_impact": {
                        "monthly_revenue_lost": round(monthly_revenue_lost, 2),
                        "total_revenue_lost": round(total_revenue_lost, 2),
                        "total_monthly_revenue": round(total_monthly_revenue, 2),
                        "monthly_loss_percentage": round(monthly_revenue_lost / total_monthly_revenue * 100, 2) if total_monthly_revenue > 0 else 0,
                        "annual_revenue_at_risk": round(monthly_revenue_lost * 12, 2)
                    },
                    "charges_distribution": charges_ranges
                }
                
            elif AsyncSessionLocal:
                # PostgreSQL queries for financial analysis
                async with AsyncSessionLocal() as session:
                    from sqlalchemy import text
                    
                    # Average charges query
                    charges_query = text("""
                        SELECT 
                            churn,
                            AVG(monthly_charges) as avg_monthly_charges,
                            AVG(total_charges) as avg_total_charges,
                            SUM(monthly_charges) as total_monthly_revenue
                        FROM customers 
                        WHERE monthly_charges IS NOT NULL AND total_charges IS NOT NULL
                        GROUP BY churn
                    """)
                    
                    # Charges distribution query
                    distribution_query = text("""
                        SELECT 
                            CASE 
                                WHEN monthly_charges < 30 THEN '0-30'
                                WHEN monthly_charges < 60 THEN '30-60'
                                WHEN monthly_charges < 90 THEN '60-90'
                                ELSE '90+'
                            END as charge_range,
                            COUNT(*) as customer_count
                        FROM customers 
                        WHERE monthly_charges IS NOT NULL
                        GROUP BY charge_range
                        ORDER BY 
                            CASE charge_range
                                WHEN '0-30' THEN 1
                                WHEN '30-60' THEN 2
                                WHEN '60-90' THEN 3
                                WHEN '90+' THEN 4
                            END
                    """)
                    
                    charges_result = await session.execute(charges_query)
                    distribution_result = await session.execute(distribution_query)
                    
                    # Process charges data
                    charges_data = {}
                    total_monthly_revenue = 0
                    monthly_revenue_lost = 0
                    
                    for row in charges_result.fetchall():
                        churn_status = "churned" if row.churn == 1 else "retained"
                        charges_data[churn_status] = {
                            "avg_monthly_charges": float(row.avg_monthly_charges or 0),
                            "avg_total_charges": float(row.avg_total_charges or 0),
                            "total_monthly_revenue": float(row.total_monthly_revenue or 0)
                        }
                        total_monthly_revenue += float(row.total_monthly_revenue or 0)
                        if row.churn == 1:
                            monthly_revenue_lost = float(row.total_monthly_revenue or 0)
                    
                    # Process distribution data
                    charges_distribution = {}
                    for row in distribution_result.fetchall():
                        charges_distribution[row.charge_range] = row.customer_count
                    
                    return {
                        "average_charges": {
                            "churned_monthly_avg": charges_data.get("churned", {}).get("avg_monthly_charges", 0),
                            "retained_monthly_avg": charges_data.get("retained", {}).get("avg_monthly_charges", 0),
                            "churned_total_avg": charges_data.get("churned", {}).get("avg_total_charges", 0),
                            "retained_total_avg": charges_data.get("retained", {}).get("avg_total_charges", 0)
                        },
                        "revenue_impact": {
                            "monthly_revenue_lost": round(monthly_revenue_lost, 2),
                            "total_monthly_revenue": round(total_monthly_revenue, 2),
                            "monthly_loss_percentage": round(monthly_revenue_lost / total_monthly_revenue * 100, 2) if total_monthly_revenue > 0 else 0,
                            "annual_revenue_at_risk": round(monthly_revenue_lost * 12, 2)
                        },
                        "charges_distribution": charges_distribution
                    }
            else:
                # Return mock data
                logger.warning("No database connection - returning mock financial data")
                return {
                    "average_charges": {
                        "churned_monthly_avg": 74.44,
                        "retained_monthly_avg": 61.27,
                        "churned_total_avg": 1531.80,
                        "retained_total_avg": 2555.34
                    },
                    "revenue_impact": {
                        "monthly_revenue_lost": 139169.36,
                        "total_revenue_lost": 2861447.32,
                        "total_monthly_revenue": 456201.85,
                        "monthly_loss_percentage": 30.51,
                        "annual_revenue_at_risk": 1670032.32
                    },
                    "charges_distribution": {
                        "0-30": 1687,
                        "30-60": 1532,
                        "60-90": 2127,
                        "90+": 1697
                    },
                    "note": "Mock data - no database connection"
                }
                
        except Exception as e:
            logger.error(f"Error calculating financial metrics: {e}")
            raise DatabaseConnectionError(f"Failed to calculate financial metrics: {str(e)}")


# Create a global instance of the analytics service
analytics_service = ChurnAnalyticsService()