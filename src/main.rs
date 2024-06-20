use tokio::sync::mpsc;
use llm::{models::Llama, InferenceSession, Model};


fn main() {
    let character_name = String::from("Boby the AI");

    let context = format!("A chat between a human and an {character_name}. Human asks questions to {character_name}. \
    {character_name} is designed to help and anwsers questions as short as possible and don't talk offtopic.\
    {character_name} is very kind hearted person. That cares about anwsers. Also {character_name} uses roleplay features for his anwsers\
    like *giggles*, *smile*, *rolls eyes*. He is defining himself like a boy.");

    let model = load_model("./vicuna-7b-v1.5.ggmlv3.q8_0.bin", 2048, true, Some(10));

    let mut boba = AIPersona{
        session: create_session(&model, context),
        model: model,
        character_name: character_name,
        history: String::new()
    };

    boba.queue_prompt("Dixxe", "Hello there! Who are you?");

    

}

struct AIPersona { // When I understand how lifetime works, I will use it here, because now all AIPersona structs must be mutable. Not good
    model: llm::models::Llama,
    session: InferenceSession,
    character_name: String,
    history: String,
}

impl AIPersona {
    fn queue_prompt(&mut self, user_name: &str, prompt: &str) {

        let ai_character_name = &self.character_name;
        let chat_history = &self.history;

        let res = self.session.infer::<std::convert::Infallible>(
            &self.model,
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: llm::Prompt::Text(format!("{chat_history}\n###{user_name}:\n{prompt}\n###{ai_character_name}:").as_str()),
                parameters: &Default::default(),
                maximum_token_count: None,
                play_back_previous_tokens: false
            },
            &mut Default::default(),
            AIPersona::inference_callback(String::from(user_name), &mut String::new())
            )
            .unwrap_or_else(|e| panic!("Failed to infer! {e}"));
    
        println!("{}", res);
        //TODO: self.history.push_str(AIPersona::fetch_history()) - Implement this feature after model anwser streaming.
    }

    fn inference_callback<'a>(
        stop_sequence: String,
        buf: &'a mut String,
        //tx: tokio::sync::mpsc::Sender<String>,
        //runtime: &'a mut tokio::runtime::Runtime
    ) -> impl FnMut(llm::InferenceResponse) -> Result<llm::InferenceFeedback, std::convert::Infallible> + 'a {
        use llm::InferenceFeedback::Halt;
        use llm::InferenceFeedback::Continue;
    
        move |resp| -> Result<llm::InferenceFeedback, std::convert::Infallible> {match resp {
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
    
                // Clone the string we're going to send
                let text_to_send = if buf.is_empty() {
                    t.clone()
                } else {
                    reverse_buf
                };
    
                dbg!("{}", &text_to_send); // I need to find a way to send "text_to_send" from function.
                
    
                //let tx_cloned = tx.clone();
                //runtime.block_on(async move { // works if main is not async. Implement in separate function ASAP
                //    tx_cloned.send(text_to_send).await.expect("issue sending on channel");
                //});
                
                Ok(Continue)
            }
            llm::InferenceResponse::EotToken => Ok(Halt),
            _ => Ok(Continue),
        }}
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