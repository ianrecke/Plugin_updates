from enum import Enum
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..scheduling.worker import get_mb


class BModel(BaseModel):
    class Config:
        use_enum_values: True
        allow_population_by_field_name = True


router = APIRouter()


# TODO: Check node type.
class PostReqModel(BModel):
    graph_json: str
    node: str


class PostRespModel(BModel):
    poll_url: str


class ResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PENDING = "PENDING"


class MarkovBlanketResult(BModel):
    blanket: list


class ResultModel(BModel):
    status: ResultStatus
    result: Optional[MarkovBlanketResult]


@router.post('/mb', response_model=PostRespModel)
def post_learn(post_req: PostReqModel):
    res = get_mb.delay(**post_req.dict())
    return PostRespModel(poll_url=res.id)


@router.get('/mb/{mb_id}', response_model=ResultModel)
def get_grn_with_id(mb_id: str):
    res = get_mb.AsyncResult(mb_id)
    if res.state == "FAILURE":
        res.forget()
        return ResultModel(status=ResultStatus.FAILURE)
    if res.state == "SUCCESS":
        result = res.get()
        res.forget()
        return ResultModel(status=ResultStatus.SUCCESS,
                           result=MarkovBlanketResult(**result))
    return ResultModel(status=ResultStatus.PENDING)
