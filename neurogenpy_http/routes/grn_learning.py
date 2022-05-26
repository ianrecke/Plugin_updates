from enum import Enum
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..scheduling.worker import learn_grn


class BModel(BaseModel):
    class Config:
        use_enum_values: True
        allow_population_by_field_name = True


router = APIRouter()


class PostReqModel(BModel):
    parcellation_id: str
    roi: str
    genes: List[str]
    algorithm: str
    estimation: str


class PostRespModel(BModel):
    poll_url: str


class ResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PENDING = "PENDING"


class NeurogenPyResult(BModel):
    result: str


class ResultModel(BModel):
    status: ResultStatus
    result: Optional[NeurogenPyResult]


@router.post('/grn', response_model=PostRespModel)
def post_learn(post_req: PostReqModel):
    res = learn_grn.delay(**post_req.dict())
    return PostRespModel(poll_url=res.id)


@router.get('/grn/{grn_id}', response_model=ResultModel)
def get_grn_with_id(grn_id: str):
    res = learn_grn.AsyncResult(grn_id)
    if res.state == "FAILURE":
        res.forget()
        return ResultModel(status=ResultStatus.FAILURE)
    if res.state == "SUCCESS":
        result = res.get()
        res.forget()
        return ResultModel(status=ResultStatus.SUCCESS,
                           result=NeurogenPyResult(**result))
    return ResultModel(status=ResultStatus.PENDING)
