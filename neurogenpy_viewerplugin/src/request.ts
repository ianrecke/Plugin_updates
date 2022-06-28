import { NEUROGENPY_ENDPOINT } from "./store";

export async function getFromNeurogenpy(subpath: string, json_body: string) {
    const res = await fetch(`${NEUROGENPY_ENDPOINT}${subpath}`, {
        method: "POST",
        body: json_body,
        headers: {
            "content-type": "application/json",
        },
    });
    if (res.status >= 400) {
        throw new Error(res.statusText);
    }
    const { poll_url } = await res.json();

    let result = await new Promise((rs, rj) => {
        const intervalRef = setInterval(async () => {
            const res = await fetch(
                `${NEUROGENPY_ENDPOINT}${subpath}/${poll_url}`
            );
            if (res.status >= 400) {
                return rj(res.statusText);
            }
            const { status, result } = await res.json();
            if (status === "SUCCESS") {
                console.log("SUCCESS", result);
                clearInterval(intervalRef);
                rs(result);
            }
            if (status === "FAILURE") {
                console.log("FAILURE");
                clearInterval(intervalRef);
                rj("operation failed");
            }
        }, 1000);
    });
    return result;
}