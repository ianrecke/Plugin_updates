from enum import Enum
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..scheduling.worker import learn_grn, get_layout, get_related, \
    check_dseparation, perform_inference


class BModel(BaseModel):
    class Config:
        use_enum_values: True
        allow_population_by_field_name = True


router = APIRouter()


class LearnPostReqModel(BModel):
    parcellation_id: str
    roi: str
    genes: List[str]
    algorithm: str
    estimation: str


class GEXFPostReqModel(BModel):
    layout: str


class RelatedNodesPostReqModel(BModel):
    node: str
    method: str


class DSepPostReqModel(BModel):
    X: list
    Y: list
    Z: list


class InferencePostReqModel(BModel):
    evidence: dict
    marginals: dict


class PostRespModel(BModel):
    poll_url: str


class ResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PENDING = "PENDING"


class LearnResult(BModel):
    gexf: str
    marginals: dict


class LearnResultModel(BModel):
    status: ResultStatus
    result: Optional[LearnResult]


class GEXFResult(BModel):
    result: dict


class GEXFResultModel(BModel):
    status: ResultStatus
    result: Optional[GEXFResult]


class RelatedNodesResult(BModel):
    result: list


class RelatedNodesResultModel(BModel):
    status: ResultStatus
    result: Optional[RelatedNodesResult]


class DSepResult(BModel):
    result: bool


class DSepResultModel(BModel):
    status: ResultStatus
    result: Optional[DSepResult]


class InferenceResult(BModel):
    marginals: dict


class InferenceResultModel(BModel):
    status: ResultStatus
    result: Optional[InferenceResult]


@router.post('/grn', response_model=PostRespModel)
def post_learn(post_req: LearnPostReqModel):
    res = learn_grn.delay(**post_req.dict())
    return PostRespModel(poll_url=res.id)


@router.get('/grn/{grn_id}', response_model=LearnResultModel)
def get_grn_with_id(grn_id: str):
    res = learn_grn.AsyncResult(grn_id)
    if res.state == "FAILURE":
        res.forget()
        return LearnResultModel(status=ResultStatus.FAILURE)
    if res.state == "SUCCESS":
        result = res.get()
        res.forget()
        return LearnResultModel(status=ResultStatus.SUCCESS,
                                result=LearnResult(**result))
    return LearnResultModel(status=ResultStatus.PENDING)


@router.post('/gexf', response_model=PostRespModel)
def post_layout(post_req: GEXFPostReqModel):
    res = get_layout.delay(**post_req.dict())
    return PostRespModel(poll_url=res.id)


@router.get('/gexf/{gexf_id}', response_model=GEXFResultModel)
def get_gexf_with_id(gexf_id: str):
    res = get_layout.AsyncResult(gexf_id)
    if res.state == "FAILURE":
        res.forget()
        return GEXFResultModel(status=ResultStatus.FAILURE)
    if res.state == "SUCCESS":
        result = res.get()
        res.forget()
        return GEXFResultModel(status=ResultStatus.SUCCESS,
                               result=GEXFResult(**result))
    return GEXFResultModel(status=ResultStatus.PENDING)


@router.post('/related', response_model=PostRespModel)
def post_related(post_req: RelatedNodesPostReqModel):
    res = get_related.delay(**post_req.dict())
    return PostRespModel(poll_url=res.id)


@router.get('/related/{related_id}', response_model=RelatedNodesResultModel)
def get_related_with_id(related_id: str):
    res = get_related.AsyncResult(related_id)
    if res.state == "FAILURE":
        res.forget()
        return RelatedNodesResultModel(status=ResultStatus.FAILURE)
    if res.state == "SUCCESS":
        result = res.get()
        res.forget()
        return RelatedNodesResultModel(status=ResultStatus.SUCCESS,
                                       result=RelatedNodesResult(**result))
    return RelatedNodesResultModel(status=ResultStatus.PENDING)


@router.post('/dseparated', response_model=PostRespModel)
def post_dseparation(post_req: DSepPostReqModel):
    res = check_dseparation.delay(**post_req.dict())
    return PostRespModel(poll_url=res.id)


@router.get('/dseparated/{dseparated_id}', response_model=DSepResultModel)
def get_dseparation_with_id(dseparated_id: str):
    res = check_dseparation.AsyncResult(dseparated_id)
    if res.state == "FAILURE":
        res.forget()
        return DSepResult(status=ResultStatus.FAILURE)
    if res.state == "SUCCESS":
        result = res.get()
        res.forget()
        return DSepResultModel(status=ResultStatus.SUCCESS,
                               result=DSepResult(**result))
    return DSepResultModel(status=ResultStatus.PENDING)


@router.post('/inference', response_model=PostRespModel)
def post_inference(post_req: InferencePostReqModel):
    res = perform_inference.delay(**post_req.dict())
    return PostRespModel(poll_url=res.id)


@router.get('/inference/{marginals_id}', response_model=InferenceResultModel)
def get_new_marginals_with_id(marginals_id: str):
    res = perform_inference.AsyncResult(marginals_id)
    if res.state == "FAILURE":
        res.forget()
        return InferenceResult(status=ResultStatus.FAILURE)
    if res.state == "SUCCESS":
        result = res.get()
        res.forget()
        return InferenceResultModel(status=ResultStatus.SUCCESS,
                                    result=InferenceResult(**result))
    return InferenceResultModel(status=ResultStatus.PENDING)
