from enum import Enum
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..scheduling.worker import learn_grn, get_layout, get_related, \
    check_dseparation, perform_inference, get_file


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
    data_type: str


class LayoutPostReqModel(BModel):
    json_bn: str
    layout: str


class IOPostReqModel(BModel):
    json_bn: str
    file_type: str


class RelatedNodesPostReqModel(BModel):
    json_bn: str
    node: str
    method: str


class DSepPostReqModel(BModel):
    json_bn: str
    X: list
    Y: list
    Z: list


class InferencePostReqModel(BModel):
    json_bn: str
    evidence: dict


class PostRespModel(BModel):
    poll_url: str


class ResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PENDING = "PENDING"


class LearnResult(BModel):
    json_bn: str
    gexf: str
    marginals: dict


class IOResult(BModel):
    result: str


class IOResultModel(BModel):
    status: ResultStatus
    result: Optional[IOResult]


class LearnResultModel(BModel):
    status: ResultStatus
    result: Optional[LearnResult]


class LayoutResult(BModel):
    result: dict


class LayoutResultModel(BModel):
    status: ResultStatus
    result: Optional[LayoutResult]


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


@router.post('/layout', response_model=PostRespModel)
def post_layout(post_req: LayoutPostReqModel):
    res = get_layout.delay(**post_req.dict())
    return PostRespModel(poll_url=res.id)


@router.get('/layout/{layout_id}', response_model=LayoutResultModel)
def get_gexf_with_id(layout_id: str):
    res = get_layout.AsyncResult(layout_id)
    if res.state == "FAILURE":
        res.forget()
        return LayoutResultModel(status=ResultStatus.FAILURE)
    if res.state == "SUCCESS":
        result = res.get()
        res.forget()
        return LayoutResultModel(status=ResultStatus.SUCCESS,
                                 result=LayoutResult(**result))
    return LayoutResultModel(status=ResultStatus.PENDING)


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
        return DSepResultModel(status=ResultStatus.FAILURE)
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
        return InferenceResultModel(status=ResultStatus.FAILURE)
    if res.state == "SUCCESS":
        result = res.get()
        res.forget()
        return InferenceResultModel(status=ResultStatus.SUCCESS,
                                    result=InferenceResult(**result))
    return InferenceResultModel(status=ResultStatus.PENDING)


@router.post('/download', response_model=PostRespModel)
def post_download(post_req: IOPostReqModel):
    res = get_file.delay(**post_req.dict())
    return PostRespModel(poll_url=res.id)


@router.get('/download/{download_id}', response_model=IOResultModel)
def download_file_with_id(download_id: str):
    res = get_file.AsyncResult(download_id)
    if res.state == "FAILURE":
        res.forget()
        return IOResultModel(status=ResultStatus.FAILURE)
    if res.state == "SUCCESS":
        result = res.get()
        res.forget()
        return IOResultModel(status=ResultStatus.SUCCESS,
                             result=IOResult(**result))
    return IOResultModel(status=ResultStatus.PENDING)
