use std::sync::{Arc, Mutex};
use llm::{models::Llama, InferenceSession, Model};
use llm::InferenceFeedback::{Halt, Continue};

fn main() {
    let character_name = String::from("Boby the helpful one");

    let context = format!("A chat between a human and an {character_name}. Human asks questions to {character_name}. \
                                {character_name} is very helpfull character and anwsers questions as short as possible and don't talk offtopic.\
                                {character_name} is very kind hearted person. That cares about anwsers. Also {character_name} uses roleplay features for his anwsers\
                                like *giggles*, *smile*, *rolls eyes*. He is defining himself like a boy.");

    let model = load_model("./vicuna-7b-v1.5.ggmlv3.q8_0.bin", 2048, true, Some(10));

    let shared_state = Arc::new(Mutex::new(String::new()));

    let mut boba = AIPersona{
        session: create_session(&model, context),
        model: model,
        character_name: character_name,
        history: String::new(),
        output: shared_state
    };

    boba.queue_prompt("Dixxe", "Hello there! What is e=mc2?");
    boba.get_result();

    boba.queue_prompt("Dixxe", "I dont't understand your previuos anwser, please repeat it.");
    boba.get_result();

}

struct AIPersona { // When I understand how lifetime works, I will use it here, because now all AIPersona structs must be mutable. Not good
    model: llm::models::Llama,
    session: InferenceSession,
    character_name: String,
    history: String,
    output: Arc<Mutex<String>>
}

impl AIPersona {
    fn queue_prompt(&mut self, user_name: &str, prompt: &str) {

        let ai_character_name = &self.character_name;
        let chat_history = &self.history;

        let res = self.session.infer::<std::convert::Infallible>(
            &self.model,
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: llm::Prompt::Text(format!("HISTORY OF CONVERSATION:{chat_history}\n###{user_name}:\n{prompt}\n###{ai_character_name}:").as_str()),
                parameters: &Default::default(),
                maximum_token_count: None,
                play_back_previous_tokens: false
            },
            &mut Default::default(),
            AIPersona::inference_callback(String::from(user_name), &mut String::new(), &self.output)
            )
            .unwrap_or_else(|e| panic!("Failed to infer! {e}"));
    
        println!("{}", res);
    }

    fn get_result(&mut self) -> String {
        
        let model_response = self.output.lock().unwrap().clone();

        self.history.push_str(&model_response.as_str()); // Model will remember context of conversation
        dbg!(&model_response);

        
        self.output.lock().unwrap().clear(); // Prepare for the next output
       
        model_response
    }

    fn inference_callback<'a>(
        stop_sequence: String,
        buf: &'a mut String,
        shared_state: &'a Arc<Mutex<String>>,
    ) -> impl FnMut(llm::InferenceResponse) -> Result<llm::InferenceFeedback, std::convert::Infallible> + 'a {
        move |resp| -> Result<llm::InferenceFeedback, std::convert::Infallible> {
            match resp {
                llm::InferenceResponse::InferredToken(t) => {
                    let mut reverse_buf = buf.clone();
                    reverse_buf.push_str(t.as_str());
                    
                    if stop_sequence.as_str().eq(reverse_buf.as_str()) {
                        buf.clear();
                        return Ok(Halt);
                    } else if stop_sequence.as_str().starts_with(reverse_buf.as_str()) {
                        buf.push_str(t.as_str());
                        return Ok(Continue);
                    }
                    
                    let text_to_send = if buf.is_empty() {
                        t.clone()
                    } else {
                        reverse_buf
                    };
                    
                    // Lock the shared state and update it
                    {
                        let mut shared_text = shared_state.lock().unwrap();
                        shared_text.push_str(text_to_send.as_str());
                    }
                    
                    Ok(Continue)
                }
                llm::InferenceResponse::EotToken => Ok(Halt),
                _ => Ok(Continue),
            }
        }
    }    
}

fn load_model(path: &str, context_size: usize, use_gpu: bool, gpu_layers: Option<usize>) -> Llama {

    let model_path = std::path::Path::new(path);

    let model = llm::load::<llm::models::Llama>(
        
        model_path,
        
        llm::TokenizerSource::Embedded,

        llm::ModelParameters { prefer_mmap: true, context_size, lora_adapters: None, use_gpu, gpu_layers, rope_overrides: None, n_gqa: None },

        llm::load_progress_callback_stdout
    )
    .unwrap_or_else(|err| panic!("Failed to load model! {err}"));

    return model
}

fn create_session(model: &llm::models::Llama, context: String) -> InferenceSession {
    let mut session = model.start_session(Default::default());

    session.feed_prompt(
        model, 
        format!("{context}").as_str(), 
        &mut Default::default(),
        llm::feed_prompt_callback(|_| {
            Ok::<llm::InferenceFeedback, std::convert::Infallible>(llm::InferenceFeedback::Continue)
        })
    )
    .expect("Failed to ingest initial prompt.");

    return session

    
}