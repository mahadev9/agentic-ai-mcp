import contextlib

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount

from mcp_servers.vehicle_insurance_claims import mcp as vehicle_mcp
from mcp_servers.healthcare_insurance_plan import mcp as healthcare_mcp


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(vehicle_mcp.session_manager.run())
        await stack.enter_async_context(healthcare_mcp.session_manager.run())
        yield


app = Starlette(
    routes=[
        Mount("/vehicle-insurance-claims", app=vehicle_mcp.streamable_http_app()),
        Mount("/healthcare-insurance-plan", app=healthcare_mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)

if __name__ == "__main__":
    uvicorn.run(app)
