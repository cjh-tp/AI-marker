const API_URL = "http://127.0.0.1:8000/mark";

const data = {
    question: "Solve for x: 3x - 7 = 14",
    student_working: "3x - 7 = 14 \n 3x = 7 \n x = 7/3", //student subtracted instead of add
    mark_scheme: "1 mark for adding 7 to both sides (3x = 21). 1 mark for correct final answer (x = 7)."
};

console.log("Sending paper to AI Marker...\n");

async function testAPI() {
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const result = await response.json();

        console.log("=== AI MARKING SUGGESTION ===");
        console.log(JSON.stringify(result, null, 2)); //prints the JSON neatly
        console.log("=============================");

    } catch (error) {
        console.error("Error:", error.message);
    }
}

testAPI();