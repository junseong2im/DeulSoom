"""
DNA Evolution Router
Migrated from app/main.py - RLHF-based DNA evolution endpoints
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import traceback
import uuid
import hashlib

from fragrance_ai.schemas.domain_models import (
    OlfactoryDNA, CreativeBrief, ScentPhenotype,
    Ingredient, NoteCategory
)
from fragrance_ai.schemas.models import (
    FragranceFormula, FormulaIngredient, ValidationLevel,
    ComplianceCheck, FormulaType
)
from fragrance_ai.services.evolution_service import get_evolution_service
from fragrance_ai.rules.ifra_rules import (
    get_ifra_checker, get_allergen_checker, ProductCategory
)
from fragrance_ai.evaluation.objectives import TotalObjective, OptimizationProfile
from fragrance_ai.utils.units import UnitConverter, MassConservationChecker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dna", tags=["DNA Evolution"])

# In-Memory Storage (for demo - use database in production)
DNA_STORAGE: Dict[str, Dict[str, Any]] = {}
EXPERIMENT_STORAGE: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class DNACreateRequest(BaseModel):
    """Request model for DNA creation"""
    brief: Dict[str, Any] = Field(..., description="Creative brief with requirements")
    name: Optional[str] = Field(None, description="Name for the DNA")
    description: Optional[str] = Field(None, description="Description")
    target_cost_per_kg: Optional[float] = Field(None, ge=0)
    product_category: str = Field("eau_de_parfum", description="Product category for IFRA")
    validation_level: ValidationLevel = ValidationLevel.STRICT


class DNACreateResponse(BaseModel):
    """Response model for DNA creation"""
    dna_id: str
    name: str
    description: Optional[str]
    ingredients: List[Dict[str, Any]]
    total_cost_per_kg: Optional[float]
    compliance: Dict[str, Any]
    created_at: str


class EvolveOptionsRequest(BaseModel):
    """Request model for evolution options"""
    dna_id: str = Field(..., description="DNA identifier")
    brief: Dict[str, Any] = Field(..., description="Creative brief")
    num_options: int = Field(3, ge=1, le=10, description="Number of variations")
    optimization_profile: OptimizationProfile = OptimizationProfile.COMMERCIAL
    algorithm: str = Field("PPO", pattern="^(PPO|REINFORCE)$")


class EvolveOptionsResponse(BaseModel):
    """Response model for evolution options"""
    experiment_id: str
    options: List[Dict[str, Any]]
    optimization_scores: Optional[Dict[str, float]]
    created_at: str


class EvolveFeedbackRequest(BaseModel):
    """Request model for evolution feedback"""
    experiment_id: str = Field(..., description="Experiment identifier")
    chosen_id: str = Field(..., description="Chosen option ID")
    rating: Optional[float] = Field(None, ge=1, le=5, description="User rating 1-5")
    notes: Optional[str] = Field(None, description="Additional feedback notes")


class EvolveFeedbackResponse(BaseModel):
    """Response model for evolution feedback"""
    status: str
    experiment_id: str
    iteration: int
    metrics: Dict[str, Any]
    message: str


class ExperimentStatusResponse(BaseModel):
    """Response model for experiment status"""
    experiment_id: str
    status: str
    created_at: str
    iterations: int
    last_feedback: Optional[Dict[str, Any]]
    dna_id: str
    algorithm: str


class ErrorResponse(BaseModel):
    """Consistent error response format"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# Helper Functions
# ============================================================================

def generate_dna_id(name: str) -> str:
    """Generate unique DNA ID"""
    timestamp = datetime.utcnow().isoformat()
    hash_input = f"{name}_{timestamp}"
    return f"dna_{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"


def create_dna_from_brief(brief: Dict[str, Any], name: str) -> OlfactoryDNA:
    """Create initial DNA from creative brief"""
    ingredients = []

    # Top notes (30%)
    if "citrus" in str(brief).lower() or brief.get("style") == "fresh":
        ingredients.append(Ingredient(
            ingredient_id="ing_001",
            name="Bergamot",
            cas_number="8007-75-8",
            concentration=15.0,
            category=NoteCategory.TOP,
            cost_per_kg=80.0,
            ifra_limit=2.0
        ))
        ingredients.append(Ingredient(
            ingredient_id="ing_002",
            name="Lemon",
            cas_number="8008-56-8",
            concentration=10.0,
            category=NoteCategory.TOP,
            cost_per_kg=60.0,
            ifra_limit=3.0
        ))

    # Heart notes (40%)
    ingredients.append(Ingredient(
        ingredient_id="ing_005",
        name="Lavender",
        cas_number="8000-28-0",
        concentration=20.0,
        category=NoteCategory.HEART,
        cost_per_kg=120.0
    ))
    ingredients.append(Ingredient(
        ingredient_id="ing_006",
        name="Geranium",
        cas_number="8000-46-2",
        concentration=20.0,
        category=NoteCategory.HEART,
        cost_per_kg=180.0
    ))

    # Base notes (30%)
    if "woody" in str(brief).lower() or brief.get("masculinity", 0) > 0.5:
        ingredients.append(Ingredient(
            ingredient_id="ing_007",
            name="Sandalwood",
            cas_number="8006-87-9",
            concentration=15.0,
            category=NoteCategory.BASE,
            cost_per_kg=400.0
        ))
        ingredients.append(Ingredient(
            ingredient_id="ing_008",
            name="Cedarwood",
            cas_number="8000-27-9",
            concentration=15.0,
            category=NoteCategory.BASE,
            cost_per_kg=80.0
        ))

    dna_id = generate_dna_id(name)

    genotype = {
        "recipe": {
            ing.ingredient_id: {
                "name": ing.name,
                "concentration": ing.concentration,
                "category": ing.category.value
            }
            for ing in ingredients
        },
        "brief_summary": str(brief)[:200]
    }

    dna = OlfactoryDNA(
        dna_id=dna_id,
        genotype=genotype,
        ingredients=ingredients,
        generation=1,
        parent_dna_ids=[]
    )

    return dna


def check_compliance(
    ingredients: List[Dict[str, Any]],
    product_category: ProductCategory
) -> Dict[str, Any]:
    """Check IFRA and allergen compliance"""

    recipe = {
        "ingredients": [
            {
                "name": ing["name"],
                "concentration": ing["concentration"]
            }
            for ing in ingredients
        ]
    }

    ifra_checker = get_ifra_checker()
    ifra_result = ifra_checker.check_ifra_violations(recipe, product_category)

    allergen_checker = get_allergen_checker()
    allergen_result = allergen_checker.check_allergens(recipe, 15.0)

    return {
        "ifra_compliant": ifra_result["compliant"],
        "ifra_violations": ifra_result["details"],
        "allergens_to_declare": allergen_result["allergens"],
        "overall_compliant": ifra_result["compliant"]
    }


# ============================================================================
# API Endpoints
# ============================================================================

@router.post(
    "/create",
    response_model=DNACreateResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_dna(request: DNACreateRequest):
    """
    Create initial DNA from creative brief

    - Generates fragrance DNA based on brief requirements
    - Checks IFRA compliance
    - Returns DNA with compliance status
    """
    try:
        if not request.brief:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error="INVALID_BRIEF",
                    message="Brief cannot be empty"
                ).model_dump()
            )

        name = request.name or f"Formula_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        dna = create_dna_from_brief(request.brief, name)

        ingredients_dict = []
        total_cost = 0.0

        for ing in dna.ingredients:
            ing_dict = {
                "ingredient_id": ing.ingredient_id,
                "name": ing.name,
                "cas_number": ing.cas_number,
                "concentration": ing.concentration,
                "percentage": ing.concentration,
                "category": ing.category.value,
                "cost_per_kg": ing.cost_per_kg,
                "ifra_limit": ing.ifra_limit
            }
            ingredients_dict.append(ing_dict)

            if ing.cost_per_kg:
                total_cost += (ing.concentration / 100) * ing.cost_per_kg

        product_category = ProductCategory(request.product_category)
        compliance = check_compliance(ingredients_dict, product_category)

        dna_data = {
            "dna_id": dna.dna_id,
            "name": name,
            "description": request.description,
            "ingredients": ingredients_dict,
            "total_cost_per_kg": total_cost if total_cost > 0 else None,
            "compliance": compliance,
            "brief": request.brief,
            "created_at": datetime.utcnow().isoformat(),
            "product_category": request.product_category,
            "validation_level": request.validation_level.value
        }
        DNA_STORAGE[dna.dna_id] = dna_data

        logger.info(f"Created DNA: {dna.dna_id} with {len(ingredients_dict)} ingredients")

        return DNACreateResponse(
            dna_id=dna.dna_id,
            name=name,
            description=request.description,
            ingredients=ingredients_dict,
            total_cost_per_kg=total_cost if total_cost > 0 else None,
            compliance=compliance,
            created_at=dna_data["created_at"]
        )

    except Exception as e:
        logger.error(f"Error creating DNA: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="INTERNAL_ERROR",
                message=f"Failed to create DNA: {str(e)}"
            ).model_dump()
        )


@router.post("/evolve/options", response_model=EvolveOptionsResponse)
async def generate_evolution_options(request: EvolveOptionsRequest):
    """
    Generate evolution options from DNA and brief

    - Uses RLHF to generate variations
    - Returns N candidate options
    - Creates experiment session for tracking
    """
    try:
        if request.dna_id not in DNA_STORAGE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    error="DNA_NOT_FOUND",
                    message=f"DNA {request.dna_id} not found"
                ).model_dump()
            )

        dna_data = DNA_STORAGE[request.dna_id]

        ingredients = []
        for ing_dict in dna_data["ingredients"]:
            ingredients.append(Ingredient(
                ingredient_id=ing_dict["ingredient_id"],
                name=ing_dict["name"],
                cas_number=ing_dict.get("cas_number"),
                concentration=ing_dict["concentration"],
                category=NoteCategory(ing_dict["category"]),
                cost_per_kg=ing_dict.get("cost_per_kg"),
                ifra_limit=ing_dict.get("ifra_limit")
            ))

        genotype = {
            "recipe": {
                ing.ingredient_id: {
                    "name": ing.name,
                    "concentration": ing.concentration,
                    "category": ing.category.value
                }
                for ing in ingredients
            },
            "brief_summary": str(dna_data.get("brief", {}))[:200]
        }

        dna = OlfactoryDNA(
            dna_id=request.dna_id,
            genotype=genotype,
            ingredients=ingredients
        )

        brief = CreativeBrief(
            brief_id=f"brief_{uuid.uuid4().hex[:8]}",
            user_id="api_user",
            **request.brief
        )

        evolution_service = get_evolution_service(algorithm=request.algorithm)

        result = evolution_service.generate_options(
            user_id="api_user",
            dna=dna,
            brief=brief,
            num_options=request.num_options
        )

        experiment_data = {
            "experiment_id": result["experiment_id"],
            "dna_id": request.dna_id,
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
            "iterations": 0,
            "algorithm": request.algorithm,
            "options": result["options"],
            "optimization_profile": request.optimization_profile.value
        }
        EXPERIMENT_STORAGE[result["experiment_id"]] = experiment_data

        formatted_options = []
        for opt in result["options"]:
            formatted_options.append({
                "id": opt["id"],
                "action": opt["action"],
                "description": opt["description"],
                "preview": {}
            })

        logger.info(f"Generated {len(formatted_options)} evolution options for experiment {result['experiment_id']}")

        return EvolveOptionsResponse(
            experiment_id=result["experiment_id"],
            options=formatted_options,
            optimization_scores=None,
            created_at=experiment_data["created_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating evolution options: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="EVOLUTION_ERROR",
                message=f"Failed to generate options: {str(e)}"
            ).model_dump()
        )


@router.post("/evolve/feedback", response_model=EvolveFeedbackResponse)
async def process_evolution_feedback(request: EvolveFeedbackRequest):
    """
    Process user feedback for RL update

    - Updates RL policy based on choice and rating
    - Advances experiment iteration
    - Returns update metrics
    """
    try:
        if request.experiment_id not in EXPERIMENT_STORAGE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    error="EXPERIMENT_NOT_FOUND",
                    message=f"Experiment {request.experiment_id} not found"
                ).model_dump()
            )

        experiment = EXPERIMENT_STORAGE[request.experiment_id]

        if experiment.get("status") != "active":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error="EXPERIMENT_INACTIVE",
                    message=f"Experiment {request.experiment_id} is not active"
                ).model_dump()
            )

        valid_ids = [opt["id"] for opt in experiment["options"]]
        if request.chosen_id not in valid_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error="INVALID_OPTION",
                    message=f"Option {request.chosen_id} not found in experiment"
                ).model_dump()
            )

        algorithm = experiment.get("algorithm", "PPO")
        evolution_service = get_evolution_service(algorithm=algorithm)

        result = evolution_service.process_feedback(
            experiment_id=request.experiment_id,
            chosen_id=request.chosen_id,
            rating=request.rating
        )

        experiment["iterations"] += 1
        experiment["last_feedback"] = {
            "chosen_id": request.chosen_id,
            "rating": request.rating,
            "notes": request.notes,
            "timestamp": datetime.utcnow().isoformat()
        }

        logger.info(
            f"Processed feedback for experiment {request.experiment_id}: "
            f"chosen={request.chosen_id}, rating={request.rating}"
        )

        return EvolveFeedbackResponse(
            status="success",
            experiment_id=request.experiment_id,
            iteration=result.get("iteration", experiment["iterations"]),
            metrics=result.get("metrics", {}),
            message="Feedback processed and policy updated"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing feedback: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="FEEDBACK_ERROR",
                message=f"Failed to process feedback: {str(e)}"
            ).model_dump()
        )


@router.get("/{dna_id}")
async def get_dna(dna_id: str):
    """Get DNA by ID"""
    try:
        if dna_id not in DNA_STORAGE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    error="DNA_NOT_FOUND",
                    message=f"DNA {dna_id} not found"
                ).model_dump()
            )

        return DNA_STORAGE[dna_id]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting DNA: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="GET_ERROR",
                message=f"Failed to get DNA: {str(e)}"
            ).model_dump()
        )


@router.get("/")
async def list_dnas(limit: int = 10, offset: int = 0):
    """List all DNAs with pagination"""
    try:
        all_dna_ids = list(DNA_STORAGE.keys())
        paginated_ids = all_dna_ids[offset:offset + limit]

        dnas = []
        for dna_id in paginated_ids:
            dna_data = DNA_STORAGE[dna_id]
            dnas.append({
                "dna_id": dna_id,
                "name": dna_data["name"],
                "created_at": dna_data["created_at"],
                "ingredient_count": len(dna_data["ingredients"]),
                "compliant": dna_data["compliance"]["overall_compliant"]
            })

        return {
            "dnas": dnas,
            "total": len(all_dna_ids),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Error listing DNAs: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="LIST_ERROR",
                message=f"Failed to list DNAs: {str(e)}"
            ).model_dump()
        )


@router.get("/experiments/{experiment_id}", response_model=ExperimentStatusResponse)
async def get_experiment_status(experiment_id: str):
    """Get experiment status and logs"""
    try:
        if experiment_id not in EXPERIMENT_STORAGE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    error="EXPERIMENT_NOT_FOUND",
                    message=f"Experiment {experiment_id} not found"
                ).model_dump()
            )

        experiment = EXPERIMENT_STORAGE[experiment_id]

        return ExperimentStatusResponse(
            experiment_id=experiment_id,
            status=experiment.get("status", "unknown"),
            created_at=experiment.get("created_at", ""),
            iterations=experiment.get("iterations", 0),
            last_feedback=experiment.get("last_feedback"),
            dna_id=experiment.get("dna_id", ""),
            algorithm=experiment.get("algorithm", "PPO")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment status: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="STATUS_ERROR",
                message=f"Failed to get experiment status: {str(e)}"
            ).model_dump()
        )


@router.delete("/experiments/{experiment_id}")
async def end_experiment(experiment_id: str):
    """End an experiment session"""
    try:
        if experiment_id not in EXPERIMENT_STORAGE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    error="EXPERIMENT_NOT_FOUND",
                    message=f"Experiment {experiment_id} not found"
                ).model_dump()
            )

        experiment = EXPERIMENT_STORAGE[experiment_id]

        algorithm = experiment.get("algorithm", "PPO")
        evolution_service = get_evolution_service(algorithm=algorithm)
        result = evolution_service.end_session(experiment_id)

        experiment["status"] = "completed"
        experiment["completed_at"] = datetime.utcnow().isoformat()

        logger.info(f"Ended experiment {experiment_id} after {experiment['iterations']} iterations")

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "total_iterations": experiment["iterations"],
            "completed_at": experiment["completed_at"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending experiment: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="END_ERROR",
                message=f"Failed to end experiment: {str(e)}"
            ).model_dump()
        )
