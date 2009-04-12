class EdgeDetection
{
	public:
		EdgeDetection();
		~EdgeDetection();

		void loadInputImage(const char* filename);
		void performEdgeDetection();
		void exportEdgeImage(const char* filename) const;
	private:
};